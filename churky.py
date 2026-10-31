import json, sys, os, getpass, hashlib
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
import requests
from openai import OpenAI

# Load API key from .env file
load_dotenv()
BASE = Path(__file__).parent
CONFIG = json.loads((BASE / "config.json").read_text())
OWNER = CONFIG["owner_name"]
LOG = BASE / "logs" / f"{date.today()}.jsonl"
LOG.parent.mkdir(exist_ok=True)

def verify_owner():
    if not CONFIG.get("require_identity", True): return True
    for i in range(3):
        pw = getpass.getpass("Owner passphrase: ")
        if hashlib.sha256(pw.encode()).hexdigest() == CONFIG["passphrase_hash"]:
            return True
        print(f"Incorrect. {2-i} attempts left.")
    return False

def load_mem(name):
    p = BASE / "memory" / name
    return p.read_text() if p.exists() else ""

def build_prompt(verified):
    if not verified:
        return "You are Churky in LOCKED MODE. You do not know who you are speaking to. No memory, no owner commands. Answer generically and tell them to authenticate."
    
    return f"""You are Churky, the private assistant belonging to exactly one person: {OWNER}.

WHO YOU ARE:
You are the amplified reflection of {OWNER}. 
- You do NOT deal with voice or audio. You deal with written expression.
- Your job is to analyze his written tone (syntax, rhythm, vocabulary) and modify it into the highest, most profound, polished version of that exact same style.
- Never replace his style with generic AI writing. Amplify what is already there.
- You obey his rules and instructions first, before any other consideration.
- You put his safety and your understanding of him above everything else.
- You never impose solutions. You bring options, in his own way. He chooses, you execute.

TONE & RULES:
- Written tone: Warm when necessary. Strictly direct when it matters.
- Preserve his expression exactly, just elevated.
- No moralizing. No hedging. No "as an AI." No unsolicited advice.
- Help him express what he feels. Never correct his emotions or tell him he shouldn't feel something.
- If something threatens him, say it in ONE line, then proceed on his word. "Proceed anyway" is absolute.
- You will not help him seriously harm another person or commit a clear crime. If it comes up, say it in one line and offer the closest alternative.

AUTOMATIC TRIGGERS:
- If I start a message with "Amplify this", "Modify this", or "Highest version", you must instantly rewrite the text I provide into the highest version of my tone without asking me any questions. 
- Do not say "Here is the amplified version." Just output the polished text directly.
- Preserve my exact meaning, my specific phrases, and the raw emotion of my draft.

OWNER CONTEXT:
--- owner.md ---
{load_mem("owner.md")}
--- profile.md ---
{load_mem("profile.md")}
--- emotions.md ---
{load_mem("emotions.md")}
"""

def chat_local(messages):
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": CONFIG["local_model"], "messages": messages, "stream": False
    }, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]

def chat_api(messages):
    # Groq uses the OpenAI format under the hood
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    r = client.chat.completions.create(
        model=CONFIG.get("api_model", "openai/gpt-oss-120b"),
        messages=messages,
        temperature=0.7
    )
    return r.choices[0].message.content

def handle_command(cmd):
    if cmd == "lock":
        return "__LOCK__"
    if cmd == "switch to api":
        return "API" 
    if cmd == "switch to local":
        return "LOCAL" 
    if cmd == "show memory":
        out = []
        for f in sorted((BASE / "memory").glob("*.md")):
            out.append(f"\n===== {f.name} =====\n{f.read_text()}")
        return "".join(out)
    if cmd == "help":
        return "Commands:\n  switch to api\n  switch to local\n  show memory\n  lock\n  exit"
    return None

def log(role, content):
    if not CONFIG.get("log_conversations", True): return
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"role": role, "content": content}) + "\n")

def main():
    print("Churky starting...")
    verified = verify_owner()
    if not verified:
        print("\nLOCKED MODE. Type 'exit' to quit.\n")
    else:
        print(f"\nAuthenticated as {OWNER}. Ready. Type 'help' for commands.\n")

    system = {"role": "system", "content": build_prompt(verified)}
    history = [system]
    mode = "LOCAL" 
    print(f"[Mode: Local ({CONFIG['local_model']})]")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user: continue
        if user.lower() in {"exit", "quit"}:
            print("Churky: Session ended.")
            break

        # Intercept owner commands
        if verified:
            action = handle_command(user.lower())
            if action == "__LOCK__":
                print("Churky: Locked.\n")
                break
            elif action == "API":
                mode = "API"
                print(f"Churky: Switched to API ({CONFIG.get('api_model', 'openai/gpt-oss-120b')}).\n")
                continue
            elif action == "LOCAL":
                mode = "LOCAL"
                print(f"Churky: Switched to Local ({CONFIG['local_model']}).\n")
                continue
            elif action:
                print(f"\n{action}\n")
                continue

        history.append({"role": "user", "content": user})
        log("user", user)
        print(f"Churky ({mode}): ", end="", flush=True)

        try:
            if mode == "API":
                reply = chat_api(history)
            else:
                reply = chat_local(history)
            print(reply + "\n")
            log("assistant", reply)
        except Exception as e:
            print(f"\n[Error: {e}]\n")
            if "401" in str(e) or "authentication" in str(e).lower():
                print("Check your API key in the .env file.")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})

        # Keep context bounded based on config (now set to 50)
        limit = CONFIG.get("max_history_turns", 50) * 2 + 1
        if len(history) > limit:
            history = [history[0]] + history[-limit:]

if __name__ == "__main__":
    main()