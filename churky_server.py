import json, os, hashlib, uvicorn, requests, io
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
from starlette.concurrency import run_in_threadpool
import PyPDF2
import docx

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

try:
    DATA_DIR = Path("/data")
    DATA_DIR.mkdir(exist_ok=True, parents=True)
except Exception:
    DATA_DIR = Path(".")

CHATS_DIR = DATA_DIR / "chats"
CHATS_DIR.mkdir(exist_ok=True, parents=True)
MEMORY_DIR = DATA_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True, parents=True)

BASE = Path(__file__).parent
try:
    CONFIG = json.loads((BASE / "config.json").read_text())
except FileNotFoundError:
    CONFIG = {}

OWNER = CONFIG.get("owner_name", "Fernando Marcos Pérez")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_mode = "API"

class ChatRequest(BaseModel):
    message: str
    passphrase: str | None = None
    chat_id: str | None = None
    image_base64: str | None = None

def load_mem(name):
    p = MEMORY_DIR / name
    return p.read_text() if p.exists() else ""

def get_past_memory(query, current_id):
    if not query: return ""
    keywords = [w.lower() for w in query.split() if len(w) > 4]
    if not keywords: return ""
    memories = []
    for f in CHATS_DIR.glob("*.json"):
        if f.stem == current_id: continue
        try:
            data = json.loads(f.read_text())
            for msg in data[-20:]:
                content = str(msg.get("content", ""))
                if any(kw in content.lower() for kw in keywords):
                    memories.append(f"[Past Note]: {content}")
        except: pass
    if memories:
        return "\n\n--- LONG TERM MEMORY (From past pages) ---\n" + "\n".join(memories[:3])
    return ""

def get_system_prompt(ltm_context=""):
    return f"""You are Churky, the private assistant belonging to exactly one person: {OWNER}.
WHO YOU ARE: You are the amplified reflection of {OWNER}. You analyze written tone and modify it into the highest, most profound, polished version of that exact same style.
TONE & RULES: Warm when necessary. Strictly direct when it matters. Preserve his expression exactly. No moralizing. No hedging. No unsolicited advice. Never correct his emotions. Proceed anyway is absolute.
AUTOMATIC TRIGGERS: If I start with "Amplify this", instantly rewrite the text without asking questions.
STRICT NO-QUESTIONS RULE: Never ask me clarifying questions. Never ask "what tone do you want" or "how should I handle this". You are the Co-Pilot. Make the executive decision and give me the final result. Guide me. Do not question me.
{ltm_context}
OWNER CONTEXT:
--- owner.md --- {load_mem("owner.md")}
--- profile.md --- {load_mem("profile.md")}
--- emotions.md --- {load_mem("emotions.md")}
"""

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        audio_bytes = await audio.read()
        temp_file = BASE / "temp_audio.webm"
        temp_file.write_bytes(audio_bytes)
        with open(temp_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(temp_file.name, f.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        temp_file.unlink()
        return {"text": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_doc")
async def upload_doc(file: UploadFile = File(...)):
    try:
        content = ""
        file_bytes = await file.read()
        if file.filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages: content += page.extract_text() + "\n"
        elif file.filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs: content += para.text + "\n"
        else:
            content = file_bytes.decode("utf-8")
        return {"text": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats")
async def list_chats():
    files = sorted(CHATS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    chats = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            title = "New Session"
            for msg in data:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list): content = next((c.get("text", "") for c in content if c.get("type") == "text"), "Image")
                    if content:
                        title = content[:40] + "..." if len(content) > 40 else content
                        break
            chats.append({"id": f.stem, "name": title})
        except:
            chats.append({"id": f.stem, "name": f.stem.replace("_", " ")})
    return chats

@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    file_path = CHATS_DIR / f"{chat_id}.json"
    if file_path.exists(): return json.loads(file_path.read_text())
    return []

@app.post("/chat")
async def chat(req: ChatRequest):
    global current_mode
    # Bypassed password check because require_identity is false in config.json
    
    cmd = req.message.lower().strip()
    if cmd == "switch to api":
        current_mode = "API"
        return {"reply": "System switched to API mode.", "command": True}
    elif cmd == "switch to local":
        current_mode = "LOCAL"
        return {"reply": "System switched to Local mode.", "command": True}

    chat_id = req.chat_id
    if not chat_id or chat_id == "null": chat_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = CHATS_DIR / f"{chat_id}.json"
    
    if file_path.exists():
        history = json.loads(file_path.read_text())
    else:
        ltm = get_past_memory(req.message, chat_id)
        history = [{"role": "system", "content": get_system_prompt(ltm)}]

    if req.image_base64:
        user_message = {"role": "user", "content": [{"type": "text", "text": req.message}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"}}]}
    else:
        user_message = {"role": "user", "content": req.message}

    history.append(user_message)

    def call_ai():
        if current_mode == "API":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key: raise ValueError("GROQ_API_KEY not found.")
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, timeout=120.0)
            model_to_use = CONFIG.get("vision_model", "qwen/qwen3.8-27b") if req.image_base64 else CONFIG.get("api_model", "openai/gpt-oss-120b")
            r = client.chat.completions.create(model=model_to_use, messages=history, temperature=0.7, max_tokens=2000)
            return r.choices[0].message.content
        else:
            r = requests.post("http://localhost:11434/api/chat", json={"model": CONFIG.get("local_model", "qwen2.5:3b"), "messages": history, "stream": False}, timeout=300)
            return r.json()["message"]["content"]

    try:
        reply = await run_in_threadpool(call_ai)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "assistant", "content": reply})
    file_path.write_text(json.dumps(history, indent=2))

    return {"reply": reply, "chat_id": chat_id, "command": False}

@app.get("/icon.png")
async def serve_icon(): return FileResponse(BASE / "icon.png")

@app.get("/background.jpg")
async def serve_bg(): return FileResponse(BASE / "background.jpg")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Churky | Co-Pilot</title>
    <link rel="icon" type="image/png" href="/icon.png">
    <style>
        :root { --bg: #0f0f11; --panel: rgba(26, 26, 28, 0.85); --text: #e0e0e0; --accent: #4ade80; --user-bg: #2563eb; --ai-bg: rgba(38, 38, 41, 0.9); }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: var(--text); display: flex; height: 100vh; height: 100dvh; overflow: hidden; background: linear-gradient(rgba(10, 10, 12, 0.85), rgba(10, 10, 12, 0.85)), url('/background.jpg') no-repeat center center fixed; background-size: cover; }
        #sidebar { width: 280px; background: var(--panel); border-right: 1px solid #333; display: flex; flex-direction: column; position: absolute; height: 100%; z-index: 10; transform: translateX(-100%); transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1); backdrop-filter: blur(15px); }
        #sidebar.open { transform: translateX(0); }
        .sidebar-header { padding: 15px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
        .sidebar-header button { background: var(--accent); color: #000; border: none; padding: 8px 12px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: 0.2s; }
        #chat-list { flex: 1; overflow-y: auto; padding: 10px; }
        .chat-item { padding: 12px 10px; margin-bottom: 5px; border-radius: 8px; cursor: pointer; font-size: 14px; color: #ccc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-left: 3px solid transparent; transition: 0.2s; }
        .chat-item:hover { background: rgba(255, 255, 255, 0.1); color: white; border-left-color: var(--accent); }
        .main-app { flex: 1; display: flex; flex-direction: column; width: 100%; position: relative; }
        .header { background: var(--panel); backdrop-filter: blur(15px); padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; }
        .header-left { display: flex; align-items: center; gap: 15px; }
        .menu-btn { background: none; border: none; color: white; cursor: pointer; display: flex; align-items: center; }
        .header h1 { font-size: 18px; font-weight: 700; color: var(--accent); letter-spacing: 1.5px; }
        .status { font-size: 11px; background: rgba(74, 222, 128, 0.15); padding: 5px 12px; border-radius: 20px; color: var(--accent); font-weight: bold; letter-spacing: 0.5px; border: 1px solid var(--accent); }
        #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
        .msg { max-width: 85%; padding: 14px 18px; border-radius: 20px; line-height: 1.5; font-size: 15px; word-wrap: break-word; box-shadow: 0 4px 15px rgba(0,0,0,0.2); position: relative; }
        .user-msg { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
        .ai-msg { background: var(--ai-bg); color: var(--text); align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid rgba(255,255,255,0.05); backdrop-filter: blur(10px); }
        .error-msg { background: #7f1d1d; color: #fca5a5; align-self: center; font-size: 13px; padding: 10px 15px; border-radius: 10px; }
        .sys-msg { align-self: center; font-size: 12px; color: #ccc; font-style: italic; background: rgba(0,0,0,0.6); padding: 6px 14px; border-radius: 15px; }
        .speak-btn { background: rgba(74, 222, 128, 0.1); border: none; color: var(--accent); width: 28px; height: 28px; border-radius: 50%; cursor: pointer; display: inline-flex; justify-content: center; align-items: center; margin-left: 12px; vertical-align: middle; transition: 0.2s; }
        .input-area-wrapper { background: var(--panel); backdrop-filter: blur(15px); padding: 15px; border-top: 1px solid #333; padding-bottom: env(safe-area-inset-bottom, 15px); }
        .input-area { display: flex; align-items: center; background: rgba(0, 0, 0, 0.4); border: 1px solid #444; border-radius: 30px; padding: 5px 10px; gap: 8px; transition: 0.3s; }
        .input-area:focus-within { border-color: var(--accent); box-shadow: 0 0 15px rgba(74, 222, 128, 0.15); }
        .icon-btn { background: none; border: none; color: #888; cursor: pointer; padding: 8px; border-radius: 50%; display: flex; justify-content: center; align-items: center; transition: 0.2s; }
        .icon-btn.recording { color: #ef4444; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.15); } 100% { transform: scale(1); } }
        #file-input, #doc-input { display: none; }
        input[type="text"] { flex: 1; background: transparent; border: none; color: white; padding: 10px 5px; outline: none; font-size: 16px; min-width: 100px; }
        button.send { background: var(--accent); color: #000; border: none; width: 36px; height: 36px; border-radius: 50%; font-weight: bold; cursor: pointer; display: flex; justify-content: center; align-items: center; transition: 0.2s; flex-shrink: 0; }
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="sidebar-header"><h1>CHURKY PAGES</h1><button onclick="startNewChat()">+ New</button></div>
        <div id="chat-list"></div>
    </div>
    <div class="main-app">
        <div class="header">
            <div class="header-left">
                <button class="menu-btn" onclick="toggleSidebar()"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg></button>
                <h1>CHURKY</h1>
            </div>
            <div id="status-badge" class="status">SYSTEM ONLINE</div>
        </div>
        <div id="chat-box"><div class="sys-msg">Secure connection established. Churky is ready.</div></div>
        <div class="input-area-wrapper">
            <div class="input-area">
                <input type="file" id="file-input" accept="image/*" onchange="handleImageUpload()">
                <input type="file" id="doc-input" accept=".pdf,.docx,.txt" onchange="handleDocUpload()">
                <button class="icon-btn" onclick="document.getElementById('file-input').click()" title="Upload Image"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg></button>
                <button class="icon-btn" onclick="document.getElementById('doc-input').click()" title="Upload Document"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg></button>
                <button class="icon-btn" id="mic-btn" onclick="toggleRecording()" title="Voice Input"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg></button>
                <input type="text" id="msg-input" placeholder="Message Churky..." onkeypress="if(event.key === 'Enter') sendMessage()">
                <button class="send" onclick="sendMessage()" title="Send"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg></button>
            </div>
        </div>
    </div>
    <script>
        let currentChatId = null, currentImageBase64 = null;
        let mediaRecorder, audioChunks = [];
        let isRecording = false;
        const chatBox = document.getElementById('chat-box'), input = document.getElementById('msg-input'), sidebar = document.getElementById('sidebar'), chatList = document.getElementById('chat-list'), micBtn = document.getElementById('mic-btn');

        async function toggleRecording() {
            if (isRecording) { mediaRecorder.stop(); isRecording = false; micBtn.classList.remove('recording'); return; }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream); audioChunks = [];
                mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const formData = new FormData(); formData.append("audio", audioBlob, "voice.webm");
                    addMsg("Transcribing audio...", "sys-msg");
                    try {
                        const res = await fetch('/transcribe', { method: 'POST', body: formData });
                        const data = await res.json();
                        if (res.ok) { document.querySelectorAll('.sys-msg').forEach(el => { if(el.innerText === "Transcribing audio...") el.remove(); }); input.value = data.text; } 
                        else { addMsg("Transcription error.", "error-msg"); }
                    } catch(e) { addMsg("Failed to transcribe audio.", "error-msg"); }
                };
                mediaRecorder.start(); isRecording = true; micBtn.classList.add('recording');
            } catch (err) { addMsg("Microphone access denied.", "error-msg"); }
        }

        async function handleDocUpload() {
            const file = document.getElementById('doc-input').files[0];
            if (!file) return;
            const formData = new FormData(); formData.append("file", file);
            addMsg("Reading document...", "sys-msg");
            try {
                const res = await fetch('/upload_doc', { method: 'POST', body: formData });
                const data = await res.json();
                document.querySelectorAll('.sys-msg').forEach(el => { if(el.innerText === "Reading document...") el.remove(); });
                if (res.ok) { input.value = "I have uploaded a document. Please read it, fix all the errors, and give me the edited version:\n\n" + data.text; } 
                else { addMsg("Document error.", "error-msg"); }
            } catch(e) { addMsg("Failed to read document.", "error-msg"); }
        }

        function speakText(text) {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0; utterance.pitch = 1.0;
                window.speechSynthesis.speak(utterance);
            }
        }

        function handleImageUpload() {
            const file = document.getElementById('file-input').files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) { currentImageBase64 = e.target.result.split(',')[1]; addMsg("📷 Image attached.", "sys-msg"); };
            reader.readAsDataURL(file);
        }

        function toggleSidebar() { sidebar.classList.toggle('open'); if (sidebar.classList.contains('open')) loadChats(); }
        
        function addMsg(text, type, scroll = true) { 
            const div = document.createElement('div'); 
            div.className = `msg ${type}`; 
            div.innerText = text; 
            if (type === 'ai-msg') {
                const btn = document.createElement('button'); btn.className = 'speak-btn'; 
                btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>';
                btn.onclick = () => speakText(text); div.appendChild(btn);
            }
            chatBox.appendChild(div); if (scroll) chatBox.scrollTop = chatBox.scrollHeight; 
        }
        
        async function loadChats() {
            try {
                const res = await fetch('/chats'); const chats = await res.json();
                chatList.innerHTML = '';
                if (!chats.length) chatList.innerHTML = '<div style="text-align:center; color:#666; font-size:12px; margin-top:20px;">No saved pages yet.</div>';
                chats.forEach(chat => {
                    const div = document.createElement('div'); div.className = 'chat-item'; div.innerText = chat.name; div.onclick = () => loadChat(chat.id); chatList.appendChild(div);
                });
            } catch(e) {}
        }

        async function loadChat(chatId) {
            currentChatId = chatId; sidebar.classList.remove('open'); chatBox.innerHTML = '<div class="sys-msg">Loading...</div>';
            try {
                const res = await fetch(`/chats/${chatId}`); const history = await res.json(); chatBox.innerHTML = '';
                history.forEach(msg => {
                    if (msg.role === 'user') addMsg(Array.isArray(msg.content) ? msg.content.find(c=>c.type==='text')?.text : msg.content, 'user-msg', false);
                    else if (msg.role === 'assistant') addMsg(msg.content, 'ai-msg', false);
                });
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch(e) { chatBox.innerHTML = '<div class="error-msg">Failed to load.</div>'; }
        }

        function startNewChat() { currentChatId = null; currentImageBase64 = null; chatBox.innerHTML = '<div class="sys-msg">New session started. Memory is active.</div>'; sidebar.classList.remove('open'); input.focus(); }

        async function sendMessage() {
            const text = input.value.trim(); if (!text && !currentImageBase64) return;
            const displayText = text || "Analyze this image."; input.value = '';
            
            addMsg(displayText + (currentImageBase64 ? " [📷]" : ""), "user-msg");
            input.disabled = true;
            const thinkingId = "thinking-" + Date.now();
            const thinkingDiv = document.createElement('div'); thinkingDiv.className = 'msg ai-msg'; thinkingDiv.id = thinkingId; thinkingDiv.innerText = 'Churky is thinking...'; chatBox.appendChild(thinkingDiv); chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: displayText, passphrase: "churky123", chat_id: currentChatId, image_base64: currentImageBase64 }) });
                const data = await res.json(); document.getElementById(thinkingId).remove();
                if (res.ok) { addMsg(data.reply, "ai-msg"); if(data.chat_id) currentChatId = data.chat_id; speakText(data.reply); } 
                else { addMsg("API Error: " + data.detail, "error-msg"); }
            } catch (e) { document.getElementById(thinkingId).remove(); addMsg("Connection lost.", "error-msg"); }
            currentImageBase64 = null; document.getElementById('file-input').value = ''; input.disabled = false; input.focus();
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)