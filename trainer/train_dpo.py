import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

from model.model_slm import SLMForCausalLM


############################################
# Initialize the configuration
############################################

if os.path.exists("./config/dpo_config.json"):
    with open('./config/dpo_config.json', 'r') as f:
        CONFIG = json.load(f)
else:
    CONFIG = None


class DPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_len: int=512):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    self.data.append(item)
                except json.JSONDecodeError:
                    continue
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def _process(self, prompt: str, answer):
        prompt_text = f"{self.tokenizer.bos_token}user\n{prompt}{self.tokenizer.eos_token}\n{self.tokenizer.bos_token}assistant\n"
        full_text = prompt_text + f"{answer}{self.tokenizer.eos_token}"

        # 分别编码
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]
        
        pad_len = self.max_len - len(full_ids)
        input_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(full_ids) + [0] * pad_len

        # prompt和padding部分为0 answer为1
        prompt_len = len(prompt_ids)
        labels_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len) + [0] * pad_len
        labels_mask = labels_mask[:self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels_mask": torch.tensor(labels_mask, dtype=torch.float)
        }
    
    def __getitem__(self, index):
        item = self.data[index]
        chosen = self._process(item['prompt'], item['chosen'])
        rejected = self._process(item['prompt'], item['rejected'])
        return {
            "chosen_ids": chosen["input_ids"],
            "chosen_mask": chosen["attention_mask"],
            "chosen_labels_mask": chosen["labels_mask"],

            "rejected_ids": rejected["input_ids"],
            "rejected_mask": rejected["attention_mask"],
            "rejected_labels_mask": rejected["labels_mask"]
        }


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    '''
        DPO loss = -log(sigmoid(beta * (log(pi_chosen / ref_chosen) - log(pi_rejected / ref_rejected))))
    '''
    # 策略模型 偏好差
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    # 参考模型 偏好差
    ref_log_ratios = ref_chosen_logps - ref_rejected_logps

    # policy偏好差 > ref偏好差
    losses = -F.logsigmoid(beta * (policy_log_ratios - ref_log_ratios))
    return losses.mean()

def get_batch_logps(model, input_ids, attention_mask, labels_mask):
    '''
        batch的log p
        labels_mask: (batch_size, seq_len), answer部分为1 只计算answer部分的loss值
    '''
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits # (batch_size, seq_len, vocab_size)

    labels = input_ids[:, 1:].clone()
    logits = logits[:, :-1, :]

    loss_mask = labels_mask[:, 1:] # 和labels一样的位移
    
    # 获取每个token的log softmax
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    # 只计算answer部分
    return (per_token_logps * loss_mask).sum(-1)



def train_dpo():
    if CONFIG is None:
        print("❌ 请检查配置文件")
        return
    
    if not os.path.exists(CONFIG["model_path"]):
        print(f"❌ 模型 {CONFIG['model_path']} 不存在, 请检查你的路径")
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    except:
        tokenizer = AutoTokenizer.from_pretrained("./out")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型加载两次，分别作为policy和ref
    policy_model = SLMForCausalLM.from_pretrained(CONFIG["model_path"]).to(device)
    ref_model = SLMForCausalLM.from_pretrained(CONFIG["model_path"]).to(device)
    ref_model.eval()
    print("✅ 模型加载完成")

    if not os.path.exists(CONFIG["data_path"]):
        print(f"❌ 文件 {CONFIG['data_path']} 不存在")
        return
    dataset = DPODataset(CONFIG["data_path"], tokenizer)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=False)

    optimizer = AdamW(policy_model.parameters(), lr=CONFIG["learning_rate"])
    policy_model.train()

    print("🔥 开始强化训练")
    for epoch in range(CONFIG["epochs"]):
        progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"epoch {epoch}")
        for batch in progress_bar:
            c_ids, c_mask = batch["chosen_ids"].to(device), batch["chosen_mask"].to(device)
            r_ids, r_mask = batch["rejected_ids"].to(device), batch["rejected_mask"].to(device)
            c_labels_mask = batch["chosen_labels_mask"].to(device)
            r_labels_mask = batch["rejected_labels_mask"].to(device)

            policy_chosen_logps = get_batch_logps(policy_model, c_ids, c_mask, c_labels_mask)
            policy_rejected_logps = get_batch_logps(policy_model, r_ids, r_mask, r_labels_mask)

            with torch.no_grad():
                ref_chosen_logps = get_batch_logps(ref_model, c_ids, c_mask, c_labels_mask)
                ref_rejected_logps = get_batch_logps(ref_model, r_ids, r_mask, r_labels_mask)
            
            loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=CONFIG["beta"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    policy_model.save_pretrained(os.path.join(CONFIG["output_dir"], "dpo_model"))
    # tokenizer.save_pretrained(CONFIG["output_dir"])
    print("🚩 完成DPO训练")