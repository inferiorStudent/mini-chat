import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW

import numpy as np
from tqdm import tqdm

from model.model_slm import SLMPretrainedModel, SLMModel, SLMForCausalLM
from model.configuration_slm import SLMConfig

class SLMForSequenceClassification(SLMPretrainedModel):
    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.num_labels = 1
        self.model = SLMModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        # 1. 获取隐状态
        outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
        hidden_states = outputs[0] # [batch, seq, dim]
        
        # 对每个token打分：output shape: [batch, seq, 1]
        scores = self.score(hidden_states)
        
        return scores.squeeze(-1)

#########################################
# ▲ 数据处理
#########################################

class PPODataset(Dataset):
    def __init__(self, prompts: list[str], tokenizer: AutoTokenizer):
        self.prompts = prompts
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index]

def collate_fn(batch_prompts, tokenizer: AutoTokenizer, device: torch.device):
    '''提示词左填充, 右对齐'''
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
    return enc.input_ids, enc.attention_mask

#########################################
# ▲ 数学计算：log probability & gae
#########################################

def get_log_probs_and_values(model, input_ids, attention_mask, critic_model=None):
    '''
    序列每个token的log_probs和values
        - log_probs: log pi_theta
        - values: V_theta
    '''
    # Actor / Ref
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    logits = outputs.logits # (batch, seq_len, vocab)
    log_probs = F.log_softmax(logits, dim=-1)

    logits_seq = logits[:, :-1, :]
    input_ids_seq = input_ids[:, 1:]

    selected_log_probs = torch.gather(logits_seq, -1, input_ids_seq.unsqueeze(-1)).squeeze(-1)

    # Critic forward
    values = None
    if critic_model is not None:
        critic_outputs = critic_model(input_ids, attention_mask=attention_mask)
        values = critic_outputs[:, :-1]
    
    return selected_log_probs, values

def compute_gae(rewards, values, bootstrap_value, gamma, var_lambda, mask):
    '''
    rewards: (batch, seq) 包含KL penalty
    values: (batch, seq)
    bootstrap_value: (batch, 1) 最后一个token之后的估值
    mask: (batch, seq) response部分为1, padding/prompt部分为0
    '''
    values_extended = torch.cat([values, bootstrap_value], dim=1)
    gae = 0
    advantages = torch.zeros_like(rewards)
    seq_len = rewards.shape[1]
    for t in reversed(range(seq_len)):
        delta = rewards[:, t] + gamma * values_extended[:, t+1] - values_extended[:, t]
        gae = delta + gamma * var_lambda * gae
        # 只计算response部分的优势, prompt相当于初始state
        advantages[:, t] = gae * mask[:, t]
    
    returns = advantages + values
    return advantages, returns



#########################################
# ▲ 训练主循环
#########################################
import os
import json

if os.path.exists("./config/ppo_config.json"):
    with open('./config/ppo_config.json', 'r') as f:
        CONFIG = json.load(f)
else:
    CONFIG = None

def train_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["sft_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ================= 加载4个网络 =================
    actor = SLMForCausalLM.from_pretrained(CONFIG["sft_model_path"]).to(device)
    ref_model = SLMForCausalLM.from_pretrained(CONFIG["sft_model_path"]).to(device)
    ref_model.eval()

    critic = SLMForSequenceClassification.from_pretrained(
        CONFIG["sft_model_path"],
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)

    # ❗❗需要单独训练, 这里仅仅作了初始化
    reward_model = SLMForSequenceClassification.from_pretrained(
        CONFIG["sft_model_path"],
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)
    reward_model.eval()

    opt_actor = AdamW(actor.parameters(), lr=CONFIG["lr_actor"])
    opt_critic = AdamW(critic.parameters(), lr=CONFIG["lr_critic"])

    dataset = PPODataset(CONFIG["prompt_data"] * 10, tokenizer) # 测试数据
    dataloader = DataLoader(dataset, batch_size=CONFIG["rollout_batch_size"], shuffle=True)

    # ================= 开始训练 =================
    total_steps = 0
    pbar = tqdm(total=CONFIG["total_episodes"])
    while total_steps < CONFIG["total_episodes"]:
        try:
            batch_prompts = next(iter(dataloader))
        except StopIteration:
            dataloader = DataLoader(dataset, batch_size=CONFIG["rollout_batch_size"], shuffle=True)
            batch_prompts = next(iter(dataloader))
        
        # rollout
        with torch.no_grad():
            prompt_ids, prompt_mask = collate_fn(batch_prompts, tokenizer, device)
            prompt_len = prompt_ids.shape[1]

            # prompt + response
            full_ids = actor.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=CONFIG["max_gen_len"],
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id
            )

            # response: 1, prompt/pad: 0. 因为batch 所以生成的时候右边会有padding
            attention_mask = (full_ids != tokenizer.pad_token_id).long()

            action_mask = torch.zeros_like(full_ids)
            action_mask[:, prompt_len:] = 1
            action_mask_seq = action_mask * attention_mask

            action_mask_seq = action_mask[:, 1:] # 和log_probs对齐

            old_log_probs, old_values = get_log_probs_and_values(actor, full_ids, attention_mask, critic)
            ref_log_probs, _ = get_log_probs_and_values(ref_model, full_ids, attention_mask, None)

            # Rewards = KL Penalty + RM Score, RM仅保存最后一个token的得分
            rm_scores = reward_model(full_ids, attention_mask=attention_mask)

            last_token_idx = attention_mask.sum(dim=1) - 1 # 最后一个非padding的idx -> shape=(batch)
            env_rewards = rm_scores[torch.arange(rm_scores.size(0)), last_token_idx]

            # KL Divergence = log_p - ref_log_p
            kl_div = old_log_probs - ref_log_probs
            rewards = - CONFIG["kl_coef"] * kl_div

            # 环境奖励只加到最后一个token上, 因为是对整个回答的评价
            last_token_idx_seq = torch.clamp(last_token_idx - 1, min=0)
            for i in range(len(env_rewards)):
                rewards[i, last_token_idx_seq[i]] += env_rewards[i]
            
            # prompt部分的reward为0
            rewards = rewards * action_mask_seq
            old_values = old_values * action_mask_seq

            # 计算GAE. bootsrap是序列结束后的价值, 通常是0或mask掉的value
            bootstrap_value = torch.zeros((old_values.shape[0], 1)).to(device)
            advantages, returns = compute_gae(rewards, old_values, bootstrap_value, CONFIG["gamma"], CONFIG["lambda"], action_mask_seq)

            # 对response部分的优势归一化, 以训练稳定
            valid_advs = advantages[action_mask_seq == 1]
            adv_mean, adv_std = valid_advs.mean(), valid_advs.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages = advantages * action_mask_seq
        
        # ================= PPO Update =================
        actor.train()
        critic.train()

        batch_size = full_ids.size(0)
        indices = np.arange(batch_size)

        # 用上面采集到的数据更新ppo_epochs次
        for _ in range(CONFIG["ppo_epochs"]):
            # 采样N个
            np.random.shuffle(indices)
            for start_idx in range(0, batch_size, CONFIG["mini_batch_size"]):
                mb_idx = indices[start_idx : start_idx + CONFIG["mini_batch_size"]]
                mb_ids = full_ids[mb_idx]
                mb_attn_mask = attention_mask[mb_idx]
                mb_action_mask = action_mask_seq[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, new_values = get_log_probs_and_values(actor, mb_ids, mb_attn_mask, critic)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CONFIG["clip_ratio"], 1.0 + CONFIG["clip_ratio"]) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).sum() / mb_action_mask.sum()

                # Critic loss = (V_new - Return)^2
                value_loss = F.mse_loss(new_values * mb_action_mask, mb_returns * mb_action_mask, reduction='sum') / mb_action_mask.sum()

                # 总
                total_loss = policy_loss + CONFIG["vf_coef"] * value_loss

                opt_actor.zero_grad()
                opt_critic.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)

                opt_actor.step()
                opt_critic.step()
        
        total_steps += batch_size
        pbar.update(batch_size)
        pbar.set_postfix({"loss": total_loss.item(), "reward": env_rewards.mean().item()})
    
    print("✅ 完成PPO")
    actor.save_pretrained(CONFIG["output_dir"])
