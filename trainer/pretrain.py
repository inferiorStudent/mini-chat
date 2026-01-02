import torch
import os
import numpy as np
import math
import json
import glob
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW

from model.model_slm import SLMForCausalLM
from model.configuration_slm import SLMConfig

############################################
# Initialize the configuration
############################################

if os.path.exists("./config/pretrain_config.json"):
    with open('./config/pretrain_config.json', 'r') as f:
        CONFIG = json.load(f)
else:
    CONFIG = None

############################################
# 读取二进制字节文件
############################################

class BinaryDataset(Dataset):
    def __init__(self, pad_token_id: int, bin_path: str, max_len: int=1024):
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        print(f"📂 加载二进制文件: {bin_path}")
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.total_size = len(self.data)
        self.num_samples = (self.total_size + max_len - 1) // max_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        start_idx = index * self.max_len
        end_idx = min(start_idx + self.max_len, self.total_size)
        chunk = self.data[start_idx:end_idx].astype(np.int64)
        
        actual_len = len(chunk)
        if actual_len < self.max_len:
            padding = np.full((self.max_len - actual_len,), self.pad_token_id, dtype=np.int64)
            chunk = np.concatenate([chunk, padding])
            
        tensor_data = torch.from_numpy(chunk)
        return {"input_ids": tensor_data, "labels": tensor_data}



def save_checkpoint(model, tokenizer, optimizer, scheduler, step: int, output_dir: str):
    '''保存模型权重 + 优化器状态 + 调度器状态'''
    save_path = os.path.join(output_dir, f"step_{step}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)

    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
    print("✅ 成功保存checkpoint")



def load_checkpoint(path, model: SLMForCausalLM, optimizer, scheduler):
    '''加载模型参数 + 调度器参数 + 优化器参数'''
    print("🔄 加载模型 + 优化器 + 调度器")
    # 保持optimizer的参数引用不变 使用load_state_dict
    pretrained_dict = SLMForCausalLM.from_pretrained(path).state_dict()
    model.load_state_dict(pretrained_dict)

    opt_path = os.path.join(path, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=torch.device("cpu")))
    else:
        print("⚠️ 未找到优化器文件")
    
    sch_path = os.path.join(path, "scheduler.pt")
    if os.path.exists(sch_path):
        scheduler.load_state_dict(torch.load(sch_path, map_location=torch.device("cpu")))
    else:
        print("⚠️ 未找到调度器文件")
    
    try:
        start_step = int(path.split("_")[-1])
    except:
        start_step = 0
    
    print(f"✅ 加载完成, ⏩ 继续进行下一步 {start_step}")
    
    return model, optimizer, scheduler, start_step



def get_dataloader(pad_token_id: int) -> DataLoader:
    '''
        首先使用 scripts.prepare_shards 将txt文件压缩为bin文件
        如果存储方式不是bin字节, 那么只需要修改这一块即可
    '''
    pending_shards = sorted(glob.glob(os.path.join(CONFIG["data_paths"], "*.bin")))
    datasets = []
    for shard_path in pending_shards:
        dataset = BinaryDataset(pad_token_id=pad_token_id, bin_path=shard_path, max_len=CONFIG["max_seq_len"])
        datasets.append(dataset)
        
    dataset = ConcatDataset(datasets)
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    print("✅ 数据加载完成")
    return dataloader



def pretrain():
    if CONFIG is None:
        print("❌ 请检查配置文件是否存在")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    model_config = SLMConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,  # 头的个数必须能够整除组的个数
        intermediate_size=2048,  # 激活函数决定: hidden_size的8/3 ~ 4
        num_key_value_heads=4,
        max_position_embeddings=CONFIG["max_seq_len"],
        tie_word_embeddings=False,
        use_moe=CONFIG["use_moe"],
        num_experts=CONFIG["num_experts"]
    )
    # 初始化model optimizer scheduler
    model = SLMForCausalLM(model_config)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    
    dataloader = get_dataloader(tokenizer.pad_token_id)
    
    total_steps = len(dataloader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    global_step = 0

    # 这里手动控制是否继续训练, 如果是的话 必须保证目录下存在safetensors或者bin权重

    # model, optimizer, scheduler, global_step = load_checkpoint(
    #     CONFIG["resume_from_checkpoint"], model, optimizer, scheduler
    # )
    # print(f"加载模型 {CONFIG['resume_from_checkpoint']}")
    # print("🔥 继续预训练")

    print("🔥 开始预训练")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    model.train()
    current_loss = 0.0

    for epoch in range(CONFIG["epochs"]):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"epoch {epoch}")
        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast(dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / CONFIG["grad_acc_steps"]
            if dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            current_loss += loss.item()

            # 每grad_acc_steps更新一次梯度
            if (step + 1) % CONFIG["grad_acc_steps"] == 0:
                # 梯度裁剪 防止MoE爆炸
                if dtype == torch.float16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % CONFIG["logging_steps"] == 0:
                    avg_loss = current_loss / CONFIG["logging_steps"]
                    ppl = math.exp(avg_loss) if avg_loss < 20 else 1e9
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", ppl=f"{ppl:.2f}")
                    current_loss = 0.0
    
    save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, CONFIG["output_dir"])
    print("🚩 训练结束, 模型已保存")