import torch
from transformers import AutoTokenizer

from model.model_slm import SLMForCausalLM

def chat():
    MODEL_PATH = "./out/dpo_model" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 加载模型 {MODEL_PATH} 到 {device}...")

    # 1. 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        tokenizer = AutoTokenizer.from_pretrained("./out")

    model = SLMForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    
    # 设为评估模式
    model.eval()
    print("✅ 模型加载成功")

    print("\n💡 输入 'exit' 退出，输入 'clear' 清空历史 (每次都是新对话)")
    print("-" * 50)

    # 3. 交互循环
    while True:
        prompt = input("\n👤 User: ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue
        
        # 显式构造对话格式, 未来在分词器文件中实现
        input_text = f"{tokenizer.bos_token}user\n{prompt}\n{tokenizer.eos_token}\n{tokenizer.bos_token}assistant\n"

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        # if "token_type_ids" in inputs:
        #     del inputs["token_type_ids"]
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                
                repetition_penalty=1.1, # 防止复读机
                
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"🤖 AI: {response}")
        break

def chat_():
    from transformers import TextIteratorStreamer
    from threading import Thread
    MODEL_PATH = "out/dpo_model" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 加载模型 {MODEL_PATH} 到 {device}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        tokenizer = AutoTokenizer.from_pretrained("./out")

    model = SLMForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    
    model.eval()
    print("✅ 模型加载成功")

    print("\n💡 输入 'exit' 退出，输入 'clear' 清空历史 (每次都是新对话)")
    print("-" * 50)

    while True:
        prompt = input("\n👤 User: ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        prompt_str = f"{tokenizer.bos_token}user\n{prompt}\n{tokenizer.eos_token}\n{tokenizer.bos_token}assistant\n"
        
        input_text = prompt_str
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512, do_sample=True, temperature=0.8, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,)

            # 必须在线程中运行 generate，否则会阻塞
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print("🤖 AI: ", end="", flush=True)
            for new_text in streamer:
                print(new_text, end="", flush=True)
            print()


def chat_origin():
    from model import origin_model_slm
    MODEL_PATH = "./out/dpo_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"🚀 加载模型 {MODEL_PATH} 到 {device}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except:
        tokenizer = AutoTokenizer.from_pretrained("./out")
    
    model = origin_model_slm.SLMForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )

    model.eval()
    prompt = "你好，"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))