import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from openai import OpenAI
from joblib import load
import subprocess
import os
import re
import jieba

#填入你的API
client = OpenAI(api_key="輸入你自己的API金鑰")

def jieba_tokenizer(text):
    return list(jieba.cut(text))

clf = load("classifier.pkl")
vectorizer = load("vectorizer.pkl")

DANGEROUS_COMMANDS = ["rm -rf", "shutdown", "reboot", "mkfs", ":(){ :|:& };:"]

def is_dangerous(cmd):
    return any(d in cmd for d in DANGEROUS_COMMANDS)

def requires_delete_confirmation(cmd):
    return any(cmd.strip().startswith(k) for k in ["rm", "unlink", "rmdir"])

def get_command_from_gpt(prompt, role="操作"):
    if role == "教學":
        system_prompt = "你是一個 Linux 教學助理，請用簡短中文說明如何執行使用者需求，包含可執行的 Linux 指令，但請先給簡要說明，教學和說明盡量不超過300字，不論任何回覆都不要用emoji符號，以免導致出錯。"
    elif role == "聊天":
        system_prompt = "你是一個友善的 Linux AI 助理，請用簡短中文回應使用者的一般聊天語句，不論任何回覆都不要用emoji符號，以免導致出錯。。"

    elif role == "模糊":
        system_prompt = "你是一個 Linux 系統助理，對於使用者的模糊問題，請簡要說明你不太能理解並提出建議，例如可以問的範例問題，不論任何回覆都不要用emoji符號，以免導致出錯。。"

    else:  
        system_prompt = "你是一個 Linux 系統助理，請將使用者的需求轉換為一行純 Bash 指令，不要說明、不要加引號或 Markdown 格式，不論任何回覆都不要用emoji符號，以免導致出錯。。"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def predict_intent(text):
    X = vectorizer.transform([text])
    intent = clf.predict(X)[0]
    confidence = max(clf.predict_proba(X)[0])
    return intent, confidence

# main
while True:
    user_input = input("請輸入你的系統操作需求（輸入 q 離開）：")
    if user_input.lower() == "q":
        break

    intent, confidence = predict_intent(user_input)
    print(f"\U0001f9e0 判斷類別：{intent}（信心值 {confidence:.2f}）")

    try:
        command = get_command_from_gpt(user_input, role=intent)

        if intent in ["聊天", "教學", "模糊"] or not re.match(r"^[\w\s\-/\.~|&]+$", command.strip()):
            print(f'💬 回覆 : "{command}"（此內容非指令，不執行）')
            continue

        if is_dangerous(command) or requires_delete_confirmation(command):
            confirm = input("⚠️ 這是危險操作，請輸入 confirm 才會執行(輸入其他文字取消)：")
            if confirm.strip() != "confirm":
                print("❌ 操作已取消。")
                continue

        result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        print("📤 執行結果：")
        print(result.stdout if result.stdout else result.stderr)

    except Exception as e:
        print("❌ 解析或執行失敗：", e)
