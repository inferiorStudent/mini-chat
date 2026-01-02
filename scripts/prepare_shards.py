import os
import glob
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

####################################
# 该脚本可以单独运行
####################################

CONFIG = {
    "source_files": [
        # "data/raw/wikipedia/enwiki-20251220-1.txt",
        # "data/raw/wikipedia/enwiki-20251220-2.txt",
        # "data/raw/wikipedia/enwiki-20251220-3.txt",
        # "data/raw/wikipedia/zhwiki-20251220-6.txt",
        # "data/raw/peoples-daily/peoples-daily-corpus.txt",
        "./dataset/temp.txt"
    ],
    "tokenizer_path": "./out",
    "output_dir": "dataset/processed",
    "shard_tokens": 100_000_000, # 1个token占2个Byte -> 1个数据文件200M
}

def process_data_into_bin():
    '''
    文本预处理: 
        将文本转换为token并编码, vocab大小为31900 因此可以用uint16来存储每个token
        由于数据文件太大, 因此将语料混合并切分为多个二进制文件
    '''
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"])

    all_lines = []
    print(" 混合源文件")
    # CONFIG["sorce_files"] = glob.glob(os.path.join("dir", "*.txt"))
    for file_path in CONFIG["source_files"]:
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过不存在的文件: {file_path}")
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if len(line.strip()) > 0]
            all_lines.extend(lines)
            print(f"   - {file_path}: {len(lines)} lines")
    
    print(f"🎲 混合打乱 {len(all_lines)} 行文本")
    random.shuffle(all_lines) # 数据量太大的话不建议这么做 因为内存不够

    print("⚙️ 分词并构建分片")
    current_token_ids = []
    shard_index = 0
    eos_id = tokenizer.eos_token_id

    batch_size = 5000
    last_batch_index = len(all_lines) // batch_size
    for i in tqdm(range(0, len(all_lines), batch_size)):
        batch = all_lines[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            current_token_ids.extend(ids)
            current_token_ids.append(eos_id)
        
        target_token_count = CONFIG["shard_tokens"]
        while len(current_token_ids) >= target_token_count:
            save_ids = current_token_ids
            current_token_ids = []
            save_path = os.path.join(CONFIG["output_dir"], f"shard_{shard_index:03d}.bin")
            print(f"💾 保存分片 {save_path}")
            arr = np.array(save_ids, dtype=np.uint16)
            with open(save_path, "wb") as f:
                f.write(arr.tobytes())
            shard_index += 1
    
    # 处理最后一坨数据
    last_batch = all_lines[last_batch_index * batch_size:]
    last_batch_encoded = tokenizer(last_batch, add_special_tokens=False)["input_ids"]
    for ids in last_batch_encoded:
        current_token_ids.extend(ids)
        current_token_ids.append(eos_id)
    if len(current_token_ids) == 0:
        return
    
    save_path = os.path.join(CONFIG["output_dir"], f"shard_{shard_index:03d}.bin")
    print(f"💾 保存最后一个分片 {save_path}")
    arr = np.array(current_token_ids, dtype=np.uint16)
    with open(save_path, "wb") as f:
        f.write(arr.tobytes())
    
    print("✅ 数据处理完成!")


if __name__ == "__main__":
    process_data_into_bin()