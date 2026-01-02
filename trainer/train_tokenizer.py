import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast, AutoTokenizer

###################################
# 该脚本可以单独运行
###################################

VOCAB_SIZE = 32000

# 替换成你自己要训练的英文语料
CORPUS_FILE_LIST = [
    # 'data/raw/wikipedia/enwiki-20251220-1.txt',
    # 'data/raw/wikipedia/enwiki-20251220-2.txt',
    # 'data/raw/wikipedia/enwiki-20251220-3.txt'
]

CHINESE_VOCAB_PATH = 'dataset/3500.txt' # 3500常用汉字

OUTPUT_DIR = './out' # 路径相对于工作路径而言
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 迭代读入文件
def data_iterator(batch_size=1000):
    for file_path in CORPUS_FILE_LIST:
        with open(file_path, 'r', encoding='utf-8') as file:
            batch = []
            for line in file:
                if len(line) == 0:
                    continue
                batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

# 直接将一个汉字作为一个token
def get_chinese_vocab(file_path: str) -> list[str]:
    chinese_vocab = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            chinese_vocab += list(set(list(line)))
    return chinese_vocab

def train_tokenizer():
    print("🚀 初始化分词器")
    chinese_vocab = get_chinese_vocab(CHINESE_VOCAB_PATH)

    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))

    # ByteLevel: GPT-2/Llama将文本转化为字节流
    # 字节流的好处就是没见到过的token不总是被识别为<unk>
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ['<unk>', '<|im_start|>', '<|im_end|>', '<pad>']
    trainer = trainers.BpeTrainer(
        vocab_size = VOCAB_SIZE - len(chinese_vocab) - 100,
        special_tokens=special_tokens,
        min_frequency=100,
        limit_alphabet=1500,
        show_progress=True,
    )

    for file_path in CORPUS_FILE_LIST:
        if not os.path.exists(file_path):
            print(f"{file_path} 不存在, 请检查拼写")
            return None

    # tokenizer.train(files=CORPUS_FILE_LIST, trainer=trainer)
    tokenizer.train_from_iterator(data_iterator(), trainer=trainer)
    print("✅ 训练完成")
    tokenizer.add_tokens(chinese_vocab)

    # 首尾自动添加特殊token
    # tokenizer.post_process = processors.TemplateProcessing(
    #     single="<s>$A</s>",
    #     pair="<s>$A</s>$B</s>",
    #     special_tokens=[
    #         ("<s>", tokenizer.token_to_id("<s>")),
    #         ("</s>", tokenizer.token_to_id("</s>"))
    #     ]
    # )

    save_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    tokenizer.save(save_path)
    print(f"✅ tokenizer.json 已经保存至 {OUTPUT_DIR}")

    print("🔄转换为Hugging Face标准")
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        tokenizer_file=save_path,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        unk_token="<unk>",
        pad_token="<pad>"
    )
    # config.json, special_tokens_map.json
    wrapped_tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ 已经完成")

    return wrapped_tokenizer

def test_tokenizer() -> None:
    print("\n🧪 测试加载与编码")
    try:
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    except:
        print("❌ 没有发现词表, 请检查是否存在")
        return
    print(f"词表大小为: {len(tokenizer)}")
    # tokenizer.size 和 len(tokenizer) 不同, 前者没有包含直接加进去的token
    text = "“很高兴见到你”的英文表达是“Nice to meet you”。"

    encoded = tokenizer.encode(text)
    print(f"编码后的结果: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"解码后的结果: {decoded}")

if __name__ == "__main__":
    # res = train_tokenizer()
    # if res is None:
    #     print(f"❌ 分词失败, 请重新尝试")
    
    test_tokenizer()