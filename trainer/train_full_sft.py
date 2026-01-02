import os
import json
import math
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

from model.model_slm import SLMForCausalLM


############################################
# Initialize the configuration
############################################

if os.path.exists("./config/sft_config.json"):
    with open('./config/sft_config.json', 'r') as f:
        CONFIG = json.load(f)
else:
    CONFIG = None

############################################

class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_len: int = 1024, format_map: dict = None):
        '''
        Args:
            format_map: 字典 {"instruction": "instruction", "input": "input", "output": "output"}
        '''
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        if format_map is None:
            self.format_map = {"instruction": "instruction", "input": "input", "output": "output"}
        else:
            self.format_map = format_map
        
        print(f"📂 加载指令微调数据集 {file_path}")
        if file_path.split('.')[-1] == "jsonl":
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue
        elif file_path.split('.')[-1] == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    self.data.append(item)
        print(f"✅ 加载了 {len(self.data)} 条样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]

        instruction = item.get(self.format_map["instruction"], "")
        input_text = item.get(self.format_map["input"], "")
        output_text = item.get(self.format_map["output"], "")

        if input_text:
            prompt_content = f"{instruction}\n{input_text}"
        else:
            prompt_content = instruction
        prompt_str = f"{self.tokenizer.bos_token}user\n{prompt_content}\n{self.tokenizer.eos_token}\n{self.tokenizer.bos_token}assistant\n"

        answer_str = f"{output_text}{self.tokenizer.eos_token}"

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_str, add_special_tokens=False)

        if len(prompt_ids) >= self.max_len - 1:
            prompt_ids = prompt_ids[:self.max_len // 2] # 直接截断
        max_answer_len = self.max_len - len(prompt_ids)
        if len(answer_ids) > max_answer_len:
            answer_ids = answer_ids[:max_answer_len]
        input_ids = prompt_ids + answer_ids

        # -100 是pytorch中忽略loss的标志
        labels = [-100] * len(prompt_ids) + answer_ids
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

############################################

def sft_train():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔄 加载分词器和模型")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    except:
        tokenizer = AutoTokenizer.from_pretrained("./out")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = SLMForCausalLM.from_pretrained(CONFIG["model_path"])
    model.to(device)

    if not os.path.exists(CONFIG["data_path"]):
        print(f"❌ 文件 {CONFIG['data_path']} 不存在，请检查")
        return

    dataset = SFTDataset(
        file_path=CONFIG["data_path"],
        tokenizer=tokenizer,
        max_len=CONFIG["max_seq_len"],
    )

    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, drop_last=True
    )
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(dataloader) * CONFIG["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    model.train()
    global_step = 0
    current_loss = 0.0
    print("🔥 开始微调")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    for epoch in range(CONFIG["epochs"]):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"processing epoch {epoch}")
        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(dtype=dtype):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / CONFIG["grad_acc_steps"]
            
            if dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            current_loss += outputs.loss.item()
            if (step + 1) % CONFIG["grad_acc_steps"] == 0:
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
                    avg_loss = current_loss / (CONFIG["logging_steps"] * CONFIG["grad_acc_steps"])
                    ppl = math.exp(avg_loss) if avg_loss < 20 else 1e9
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", ppl=f"{ppl:.2f}")
                    current_loss = 0.0
                
    final_path = os.path.join(CONFIG["output_dir"], "sft_model")
    model.save_pretrained(final_path)
    print(f"🚩 完成指令微调, 模型已保存至于 {final_path}")