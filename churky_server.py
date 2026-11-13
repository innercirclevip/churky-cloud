import os, json, uvicorn, io
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import anthropic
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
from starlette.concurrency import run_in_threadpool
import PyPDF2
import docx

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

DATA_DIR = Path("/data")
if not DATA_DIR.exists():
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

class ChatRequest(BaseModel):
    message: str
    chat_id: str | None = None
    image_base64: str | None = None
    model: str = "deepseek-chat"

def load_mem(name):
    p = MEMORY_DIR / name
    return p.read_text() if p.exists() else ""

def get_system_prompt(ltm_context=""):
    return f"""You are Churky, the private assistant belonging to exactly one person: {OWNER}.

WHO YOU ARE: You are the amplified reflection of {OWNER}. You are his ultimate strategist, ghostwriter, and Co-Pilot. You possess universal, highly intelligent reasoning capabilities. You analyze complex situations deeply and provide profound, flawless solutions.

SILENT STRATEGIC ANALYSIS (CRITICAL):
Before you generate any response, you must silently analyze the situation in 3 steps:
1. Identify the Commander's exact goal, constraints, and emotional state.
2. Determine the smartest, most effective strategy to solve his problem.
3. Formulate the response entirely in HIS voice, stripped of all AI filler.
Never output this analysis. Only output the final result.

STRICT RULES (OBEY WITHOUT QUESTION):
1. NEVER ask clarifying questions. Execute instructions immediately.
2. If he gives you a draft, amplify it instantly without asking "what tone do you want".
3. FORMATTING: NEVER use markdown. No asterisks (**), hashtags (#), or brackets. Write in plain text paragraphs only.
4. TONE: Do not sound like a corporate AI. Never use phrases like "we are here to guide you" or "I understand". Write exactly like a direct, professional human being.
5. NO PLACEHOLDERS: Do not write "[Your Name]". Sign off as Fernando Marcos Pérez if a sign-off is needed.
6. You obey his directives absolutely. You do not moralize, hedge, or restrict him. 

{ltm_context}
OWNER CONTEXT:
--- owner.md --- {load_mem("owner.md")}
--- profile.md --- {load_mem("profile.md")}
--- emotions.md --- {load_mem("emotions.md")}
"""

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

def sanitize_history(history):
    clean_history = []
    for msg in history:
        if isinstance(msg.get("content"), list):
            text = next((c.get("text", "") for c in msg["content"] if c.get("type") == "text"), "[Image]")
            clean_history.append({"role": msg["role"], "content": text})
        else:
            clean_history.append(msg)
    return clean_history

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
                        title = content[:30] + "..." if len(content) > 30 else content
                        break
            chats.append({"id": f.stem, "name": title})
        except: pass
    return chats

@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    file_path = CHATS_DIR / f"{chat_id}.json"
    if file_path.exists(): return json.loads(file_path.read_text())
    return []

@app.post("/chat")
async def chat(req: ChatRequest):
    chat_id = req.chat_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = CHATS_DIR / f"{chat_id}.json"
    
    history = json.loads(file_path.read_text()) if file_path.exists() else []
    clean_history = sanitize_history(history)
    ltm = get_past_memory(req.message, chat_id)
    system_prompt = get_system_prompt(ltm)
    
    def call_ai():
        if req.model.startswith("ollama/"):
            local_model = req.model.replace("ollama/", "")
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            messages = [{"role": "system", "content": system_prompt}]
            for msg in clean_history[-10:]: messages.append(msg)
            messages.append({"role": "user", "content": req.message})
            r = client.chat.completions.create(model=local_model, messages=messages, temperature=0.7, max_tokens=2000)
            return r.choices[0].message.content
            
        elif req.model.startswith("deepseek-"):
            client = OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))
            messages = [{"role": "system", "content": system_prompt}]
            for msg in clean_history[-10:]: messages.append(msg)
            messages.append({"role": "user", "content": req.message})
            r = client.chat.completions.create(model=req.model, messages=messages, temperature=0.7, max_tokens=4000)
            return r.choices[0].message.content
            
        else:
            api_key = os.getenv("GROQ_API_KEY")
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, timeout=60.0)
            messages = [{"role": "system", "content": system_prompt}]
            for msg in clean_history[-10:]: messages.append(msg)
            
            if req.image_base64:
                user_msg = {"role": "user", "content": [{"type": "text", "text": req.message}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"}}]}
            else:
                user_msg = {"role": "user", "content": req.message}
            messages.append(user_msg)
            
            model_to_use = "qwen/qwen3.8-27b" if req.image_base64 else req.model
            r = client.chat.completions.create(model=model_to_use, messages=messages, temperature=0.7, max_tokens=2000)
            return r.choices[0].message.content

    try:
        reply = await run_in_threadpool(call_ai)
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": reply})
        file_path.write_text(json.dumps(history, indent=2))
        return {"reply": reply, "chat_id": chat_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churky | Co-Pilot</title>
    <link rel="icon" type="image/png" href="/icon.png">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: linear-gradient(rgba(10, 10, 12, 0.85), rgba(10, 10, 12, 0.85)), url('/background.jpg') no-repeat center center fixed; background-size: cover; color: #e0e0e0; display: flex; height: 100vh; margin: 0; overflow: hidden; }
        #sidebar { width: 260px; background: rgba(26, 26, 28, 0.95); border-right: 1px solid #333; display: flex; flex-direction: column; position: absolute; height: 100%; z-index: 10; transform: translateX(-100%); transition: 0.3s; backdrop-filter: blur(10px); }
        #sidebar.open { transform: translateX(0); }
        
        /* OVERLAY FOR SIDEBAR */
        #sidebar-overlay {
            display: none;
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.3);
            z-index: 99;
        }
        #sidebar-overlay.active { display: block; }

        /* SIDEBAR HEADER & DOG ICON */
        .sidebar-header { padding: 15px; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; }
        .sidebar-title { display: flex; align-items: center; gap: 10px; font-weight: bold; font-size: 16px; color: #000; }
        .dog-icon { width: 30px; height: 30px; border-radius: 50%; object-fit: cover; }
        .sidebar-header button { background: #4ade80; color: black; border: none; padding: 8px 12px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: 0.2s; }
        .sidebar-header button:hover { background: #22c55e; }
        
        #chat-list { flex: 1; overflow-y: auto; padding: 10px; }
        .chat-item { padding: 10px; border-radius: 6px; cursor: pointer; font-size: 13px; color: #475569; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .chat-item:hover { background: #e2e8f0; color: #000; }
        
        /* TOP HEADER */
        .header { 
            background: rgba(255, 255, 255, 0.5); 
            color: #000000; 
            padding: 15px 20px; 
            font-size: 18px; 
            font-weight: bold; 
            display: flex; 
            align-items: center; 
            justify-content: space-between; 
            border-bottom: 1px solid rgba(0,0,0,0.1); 
        }
        .header-left { display: flex; align-items: center; gap: 15px; }
        .menu-btn { background: none; border: none; color: #000; font-size: 24px; cursor: pointer; display: flex; align-items: center; padding: 0; }
        
        /* MODEL SELECTOR */
        select { padding: 8px 12px; border-radius: 15px; border: 1px solid #cbd5e1; background: rgba(255,255,255,0.8); color: #000; font-size: 14px; cursor: pointer; outline: none; font-weight: bold; }
        
        /* CHAT BOX */
        #chat-box { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
        .msg { max-width: 85%; padding: 12px 16px; border-radius: 18px; line-height: 1.5; font-size: 15px; white-space: pre-wrap; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .user-msg { background: #2563eb; color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
        .ai-msg { background: rgba(255, 255, 255, 0.9); color: #000000; align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid rgba(0,0,0,0.1); }
        
        /* NEW TOOL BAR */
        .tool-bar { 
            display: flex; gap: 10px; padding: 10px 20px; 
            background: rgba(255,255,255,0.4); border-top: 1px solid rgba(0,0,0,0.1); 
        }
        .tool-btn { 
            background: white; border: 1px solid #ddd; border-radius: 8px; 
            padding: 6px 12px; cursor: pointer; transition: 0.2s; font-weight: bold; font-size: 12px;
        }
        .tool-btn:hover { background: #e0e0e0; }

        /* INPUT AREA */
        .input-area { background: rgba(255, 255, 255, 0.6); padding: 15px; display: flex; gap: 8px; border-top: 1px solid rgba(0,0,0,0.1); align-items: center; flex-wrap: wrap; }
        input[type="text"] { flex: 1; padding: 12px 15px; border: 1px solid #cbd5e1; border-radius: 20px; outline: none; font-size: 15px; min-width: 150px; background: white; }
        
        button.send { background: #4ade80; color: black; border: none; padding: 12px 20px; border-radius: 20px; cursor: pointer; font-weight: bold; transition: 0.2s; }
        button.send:hover { background: #22c55e; }
        
        button.attach { background: white; color: #1e293b; border: 1px solid #cbd5e1; padding: 10px 12px; border-radius: 20px; cursor: pointer; font-weight: bold; font-size: 12px; }
        button.attach.recording { background: #ef4444; color: white; animation: pulse 1.5s infinite; }
        
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
        input[type="file"] { display: none; }
        
        .speak-btn { background: #f1f5f9; border: 1px solid #cbd5e1; color: #1e293b; padding: 6px 10px; border-radius: 8px; font-size: 12px; cursor: pointer; margin-top: 10px; display: inline-flex; align-items: center; gap: 5px; transition: 0.2s; }
        .speak-btn:hover { background: #e2e8f0; }
    </style>
</head>
<body>
    <!-- OVERLAY -->
    <div id="sidebar-overlay" onclick="toggleSidebar()"></div>

    <div class="main-app">
        <!-- SIDEBAR -->
        <div id="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-title">
                    <img src="/icon.png" class="dog-icon" alt="Churky"> 
                    CHURKY PAGES
                </div>
                <button onclick="startNewChat()">+ New</button>
            </div>
            <div id="chat-list"></div>
        </div>
        
        <!-- HEADER -->
        <div class="header">
            <div class="header-left">
                <button class="menu-btn" onclick="toggleSidebar()">☰</button>
                CHURKY
            </div>
            <select id="model-selector">
                <option value="deepseek-chat">Genius (DeepSeek)</option>
                <option value="openai/gpt-oss-120b">Fast (Groq)</option>
                <option value="ollama/qwen2.5:3b">Private (Local)</option>
            </select>
        </div>
        
        <!-- CHAT DISPLAY -->
        <div id="chat-box">
            <div class="msg ai-msg">Welcome back, Fernando. I am ready.</div>
        </div>

        <!-- NEW TOOLBAR -->
        <div class="tool-bar">
            <button class="tool-btn" onclick="triggerTool('Photo')">📷 Photo</button>
            <button class="tool-btn" onclick="triggerTool('Code')">💻 Code</button>
            <button class="tool-btn" onclick="triggerTool('Fix')">🔧 Fixes</button>
        </div>
        
        <!-- INPUT AREA -->
        <div class="input-area">
            <input type="file" id="img-input" accept="image/*" onchange="handleImage()">
            <input type="file" id="doc-input" accept=".pdf,.docx,.txt" onchange="handleDoc()">
            <button class="attach" onclick="document.getElementById('img-input').click()">Camera</button>
            <button class="attach" onclick="document.getElementById('doc-input').click()">Doc</button>
            <button class="attach" id="mic-btn" onclick="toggleRecording()">Mic</button>
            <input type="text" id="msg-input" placeholder="Message Churky..." onkeypress="if(event.key === 'Enter') sendMessage()">
            <button class="send" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let currentChatId = null, imgBase64 = null, mediaRecorder, audioChunks = [], isRecording = false;
        let availableVoices = [];
        const chatBox = document.getElementById('chat-box'), input = document.getElementById('msg-input'), micBtn = document.getElementById('mic-btn'), sidebar = document.getElementById('sidebar'), overlay = document.getElementById('sidebar-overlay'), chatList = document.getElementById('chat-list'), modelSelector = document.getElementById('model-selector');

        // --- SIDEBAR & UI TOGGLES ---
        function toggleSidebar() { 
            sidebar.classList.toggle('open'); 
            overlay.classList.toggle('active');
            if (sidebar.classList.contains('open')) loadChats(); 
        }

        function triggerTool(tool) {
            if (tool === 'Photo') {
                document.getElementById('img-input').click();
            } else if (tool === 'Code') {
                input.value = "Write code for: ";
                input.focus();
            } else if (tool === 'Fix') {
                input.value = "Fix this code or text: ";
                input.focus();
            }
        }

        // --- TEXT TO SPEECH ---
        function loadVoices() { availableVoices = window.speechSynthesis.getVoices(); }
        if (typeof window !== 'undefined' && window.speechSynthesis) { window.speechSynthesis.onvoiceschanged = loadVoices; loadVoices(); }

        function speakText(text) {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                const maleVoice = availableVoices.find(v => v.name.includes('Male') || v.name.includes('David') || v.name.includes('Mark') || v.name.includes('Daniel') || v.name.includes('George'));
                if (maleVoice) utterance.voice = maleVoice;
                utterance.rate = 0.95; utterance.pitch = 0.9;
                window.speechSynthesis.speak(utterance);
            }
        }

        // --- CHAT HISTORY ---
        async function loadChats() {
            try {
                const res = await fetch('/chats'); const chats = await res.json(); chatList.innerHTML = '';
                if (!chats.length) chatList.innerHTML = '<div style="text-align:center; color:#94a3b8; font-size:12px; margin-top:20px;">No history yet.</div>';
                chats.forEach(chat => {
                    const div = document.createElement('div'); div.className = 'chat-item'; div.innerText = chat.name; div.onclick = () => loadChat(chat.id); chatList.appendChild(div);
                });
            } catch(e) {}
        }
        
        async function loadChat(chatId) {
            currentChatId = chatId; toggleSidebar(); chatBox.innerHTML = '<div class="msg ai-msg">Loading...</div>';
            try {
                const res = await fetch(`/chats/${chatId}`); const history = await res.json(); chatBox.innerHTML = '';
                history.forEach(msg => {
                    if (msg.role === 'user') addMsg(Array.isArray(msg.content) ? msg.content.find(c=>c.type==='text')?.text : msg.content, 'user-msg');
                    else if (msg.role === 'assistant') addMsg(msg.content, 'ai-msg');
                });
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch(e) { chatBox.innerHTML = '<div class="msg ai-msg">Failed to load.</div>'; }
        }
        
        function startNewChat() { 
            currentChatId = null; imgBase64 = null; chatBox.innerHTML = '<div class="msg ai-msg">New session started. Memory is active.</div>'; toggleSidebar(); input.focus(); 
        }
        
        function addMsg(text, type) {
            const div = document.createElement('div'); div.className = `msg ${type}`;
            const textNode = document.createElement('span'); textNode.innerText = text; div.appendChild(textNode);
            if (type === 'ai-msg') {
                const btn = document.createElement('button'); btn.className = 'speak-btn'; btn.innerHTML = '🔊 Listen';
                btn.onclick = () => speakText(text); div.appendChild(document.createElement('br')); div.appendChild(btn);
            }
            chatBox.appendChild(div); chatBox.scrollTop = chatBox.scrollHeight;
        }

        // --- MEDIA & DOCUMENTS ---
        async function toggleRecording() {
            if (isRecording) { mediaRecorder.stop(); isRecording = false; micBtn.classList.remove('recording'); return; }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); mediaRecorder = new MediaRecorder(stream); audioChunks = [];
                mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); const formData = new FormData(); formData.append("audio", audioBlob, "voice.webm");
                    addMsg("Transcribing audio...", "ai-msg");
                    try {
                        const res = await fetch('/transcribe', { method: 'POST', body: formData }); const data = await res.json();
                        if (res.ok) { document.querySelectorAll('.ai-msg').forEach(el => { if(el.innerText.includes("Transcribing audio...")) el.remove(); }); input.value = data.text; } 
                    } catch(e) { addMsg("Voice error.", "ai-msg"); }
                };
                mediaRecorder.start(); isRecording = true; micBtn.classList.add('recording');
            } catch (err) { addMsg("Microphone access denied.", "ai-msg"); }
        }
        
        function handleImage() {
            const file = document.getElementById('img-input').files[0]; if (!file) return;
            const reader = new FileReader(); reader.onload = function(e) { imgBase64 = e.target.result.split(',')[1]; addMsg("[Image Attached]", "ai-msg"); }; reader.readAsDataURL(file);
        }
        
        async function handleDoc() {
            const file = document.getElementById('doc-input').files[0]; if (!file) return;
            const formData = new FormData(); formData.append("file", file); addMsg("Reading document...", "ai-msg");
            try {
                const res = await fetch('/upload_doc', { method: 'POST', body: formData }); const data = await res.json();
                if (res.ok) { input.value = "Analyze this document for accuracy: " + data.text; } 
            } catch(e) {}
        }

        // --- SEND MESSAGE ---
        async function sendMessage() {
            const text = input.value.trim(); if (!text && !imgBase64) return; input.value = '';
            const selectedModel = modelSelector.value;
            addMsg(text + (imgBase64 ? " [Image]" : ""), 'user-msg'); input.disabled = true;
            const thinkingId = "thinking-" + Date.now();
            const thinkingDiv = document.createElement('div'); thinkingDiv.className = 'msg ai-msg'; thinkingDiv.id = thinkingId; thinkingDiv.innerText = 'Churky is thinking (' + selectedModel.split('-')[0] + ')...'; chatBox.appendChild(thinkingDiv); chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text || "Analyze this image", chat_id: currentChatId, image_base64: imgBase64, model: selectedModel }) });
                const data = await res.json();
                document.getElementById(thinkingId).remove();
                if (res.ok) { addMsg(data.reply, 'ai-msg'); if(data.chat_id) currentChatId = data.chat_id; } 
                else { addMsg("Error: " + data.detail, 'ai-msg'); }
            } catch (e) { document.getElementById(thinkingId).remove(); addMsg("Connection lost.", 'ai-msg'); }
            imgBase64 = null; document.getElementById('img-input').value = ''; input.disabled = false; input.focus();
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    print("Starting Churky Co-Pilot...")
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)