import sys, torch, os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).resolve().parent
QWEN_DIR = BASE_DIR / "model" / "qwen_model"

def main():
    if len(sys.argv) < 2:
        print("抱歉，我暂时无法回答这个问题。")
        return

    question = sys.argv[1].strip()
    if not question:
        print("抱歉，我暂时无法回答这个问题。")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN_DIR,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            QWEN_DIR,
            local_files_only=True,
            torch_dtype="auto",
            trust_remote_code=False,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        messages = [
            {"role": "system", "content": "你是一个智能计算系统课程的AI助手。"},
            {"role": "user", "content": question},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

        prompt_len = inputs["input_ids"].shape[1]
        answer = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

        if answer:
            print(answer)
        else:
            print("抱歉，我暂时无法完整回答这个问题。")

    except Exception:
        print("抱歉，我暂时无法回答这个问题。")

if __name__ == "__main__":
    main()
