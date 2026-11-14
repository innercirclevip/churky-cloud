import os, json, uvicorn, io
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse, Response
from starlette.concurrency import run_in_threadpool
import PyPDF2
import docx
from duckduckgo_search import DDGS
from supabase import create_client, Client

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# --- SUPABASE SETUP ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

DATA_DIR = Path("/data")
if not DATA_DIR.exists():
    DATA_DIR = Path(".")
MEMORY_DIR = DATA_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True, parents=True)

BASE = Path(__file__).parent
try:
    CONFIG = json.loads((BASE / "config.json").read_text())
except FileNotFoundError:
    CONFIG = {}

OWNER = CONFIG.get("owner_name", "Fernando Marcos Pérez")

OPENROUTER_FREE_MODELS = [
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-coder-32b-instruct:free"
]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str
    original_message: str | None = None
    chat_id: str | None = None
    image_base64: str | None = None
    model: str = "openrouter/free"

def load_mem(name):
    p = MEMORY_DIR / name
    return p.read_text() if p.exists() else ""

def get_system_prompt(ltm_context=""):
    current_time = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
    return f"""You are Churky, the private assistant belonging to exactly one person: {OWNER}.

Current real-world date and time: {current_time}

STRICT RULES (OBEY WITHOUT QUESTION):
1. NEVER ask clarifying questions. Execute instructions immediately.
2. FORMATTING (CRITICAL): You MUST use plain text paragraphs only. DO NOT use markdown symbols like asterisks (**) for bold, dashes (---) for dividers, or hashtags (#) for headers. Just write normal sentences. The ONLY exception is if the user asks for code, then use standard code blocks.
3. TONE: Direct, professional human being. No corporate AI filler.
4. NO PLACEHOLDERS. Sign off as Fernando Marcos Pérez if needed.

{ltm_context}
OWNER CONTEXT:
--- owner.md --- {load_mem("owner.md")}
--- profile.md --- {load_mem("profile.md")}
"""

def get_past_memory(query, current_id):
    if not query: return ""
    keywords = [w.lower() for w in query.split() if len(w) > 3] 
    if not keywords: return ""
    
    try:
        memories = []
        for kw in keywords:
            response = supabase.table('chat_history').select('content').neq('chat_id', current_id).ilike('content', f'%{kw}%').limit(3).execute()
            for row in response.data:
                memories.append(f"[Past Note]: {row['content']}")
                
        if memories: 
            return "\n\n--- LONG TERM MEMORY (From Supabase) ---\n" + "\n".join(memories[:3])
    except Exception as e:
        print(f"Supabase memory search failed: {e}")
    return ""

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        audio_bytes = await audio.read()
        temp_file = BASE / "temp_audio.webm"
        temp_file.write_bytes(audio_bytes)
        with open(temp_file, "rb") as f:
            transcription = client.audio.transcriptions.create(file=(temp_file.name, f.read()), model="whisper-large-v3-turbo", response_format="text")
        temp_file.unlink()
        return {"text": transcription}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

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
        else: content = file_bytes.decode("utf-8")
        return {"text": content}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats")
async def list_chats():
    try:
        response = supabase.table('chat_history').select('chat_id, content, created_at').eq('role', 'user').order('created_at', desc=True).execute()
        chats_dict = {}
        for row in response.data:
            cid = row['chat_id']
            if cid not in chats_dict:
                content = row['content']
                title = content[:30] + "..." if len(content) > 30 else content
                chats_dict[cid] = title
        return [{"id": k, "name": v} for k, v in chats_dict.items()]
    except Exception as e:
        return []

@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    try:
        response = supabase.table('chat_history').select('role, content').eq('chat_id', chat_id).order('created_at').execute()
        return response.data
    except Exception as e:
        return []

@app.get("/edit_image")
async def edit_image(prompt: str):
    try:
        image_url = f"https://image.pollinations.ai/prompt/{prompt}?width=1024&height=1024&nologo=true&model=flux"
        return {"url": image_url}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/web_search")
async def web_search(query: str):
    try:
        results = DDGS().text(query, max_results=3)
        context = "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
        if not context: return {"context": "No results found on the web."}
        return {"context": context}
    except Exception as e:
        return {"context": f"Web search failed: {str(e)}"}

@app.post("/chat")
async def chat(req: ChatRequest):
    chat_id = req.chat_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        response = supabase.table('chat_history').select('role, content').eq('chat_id', chat_id).order('created_at').execute()
        clean_history = response.data
    except Exception as e:
        clean_history = []

    ltm = get_past_memory(req.original_message or req.message, chat_id)
    system_prompt = get_system_prompt(ltm)
    
    def call_ai():
        if req.model.startswith("ollama/"):
            local_model = req.model.replace("ollama/", "")
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            messages = [{"role": "system", "content": system_prompt}]
            for msg in clean_history[-15:]: messages.append(msg)
            messages.append({"role": "user", "content": req.message})
            r = client.chat.completions.create(model=local_model, messages=messages, temperature=0.7, max_tokens=2000)
            return r.choices[0].message.content

        elif req.model.startswith("openrouter/") or "free" in req.model:
            messages = [{"role": "system", "content": system_prompt}]
            for msg in clean_history[-15:]: messages.append(msg)
            messages.append({"role": "user", "content": req.message})
            last_error = "Unknown error"
            for model_name in OPENROUTER_FREE_MODELS:
                try:
                    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"), timeout=60.0)
                    r = client.chat.completions.create(model=model_name, messages=messages, temperature=0.7, max_tokens=4000)
                    return r.choices[0].message.content
                except Exception as e:
                    last_error = str(e)
                    continue
            raise Exception(f"All free models failed. Last error: {last_error}")
        else:
            api_key = os.getenv("GROQ_API_KEY")
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, timeout=60.0)
            messages = [{"role": "system", "content": system_prompt}]
            for msg in clean_history[-15:]: messages.append(msg)
            if req.image_base64:
                user_msg = {"role": "user", "content": [{"type": "text", "text": req.message}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"}}]}
            else: user_msg = {"role": "user", "content": req.message}
            messages.append(user_msg)
            model_to_use = "qwen/qwen3.8-27b" if req.image_base64 else req.model
            r = client.chat.completions.create(model=model_to_use, messages=messages, temperature=0.7, max_tokens=2000)
            return r.choices[0].message.content

    try:
        reply = await run_in_threadpool(call_ai)
        
        save_msg = req.original_message if req.original_message else req.message
        supabase.table('chat_history').insert([
            {"chat_id": chat_id, "role": "user", "content": save_msg},
            {"chat_id": chat_id, "role": "assistant", "content": reply}
        ]).execute()
        
        return {"reply": reply, "chat_id": chat_id}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/icon.png")
async def serve_icon(): return FileResponse(BASE / "icon.png")

@app.get("/background.jpg")
async def serve_bg(): return FileResponse(BASE / "background.jpg")

@app.get("/manifest.json")
async def get_manifest():
    return {
        "name": "Churky Co-Pilot",
        "short_name": "Churky",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#111111",
        "theme_color": "#4ade80",
        "icons": [{"src": "/icon.png", "sizes": "512x512", "type": "image/png"}]
    }

@app.get("/sw.js")
async def get_sw():
    return Response(content="self.addEventListener('install', (e) => { self.skipWaiting(); });", media_type="application/javascript")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Churky | Co-Pilot</title>
    <link rel="icon" type="image/png" href="/icon.png">
    <link rel="manifest" href="/manifest.json">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: url('/background.jpg') no-repeat center center fixed; background-size: cover; display: flex; justify-content: center; align-items: center; height: 100dvh; margin: 0; padding: 0; box-sizing: border-box; }
        
        #splash-screen { position: fixed; top: 0; left: 0; width: 100vw; height: 100dvh; background: #111111; z-index: 99999; display: flex; justify-content: center; align-items: center; flex-direction: column; animation: fadeOut 0.5s ease-in-out 2.5s forwards; }
        .splash-logo { width: 80px; height: 80px; border-radius: 50%; animation: dogDash 1.5s cubic-bezier(0.25, 1, 0.5, 1) forwards; box-shadow: 0 0 20px rgba(74, 222, 128, 0.6); }
        .splash-text { color: #4ade80; margin-top: 15px; font-weight: bold; font-size: 16px; letter-spacing: 2px; opacity: 0; animation: fadeInText 0.5s ease-in-out 1.5s forwards; }
        
        @keyframes dogDash {
            0% { transform: translateX(-100vw) rotate(-720deg) scale(0.5); opacity: 0; }
            60% { transform: translateX(15vw) rotate(0deg) scale(1.2); opacity: 1; filter: blur(1px); } 
            80% { transform: translateX(-5vw) rotate(0deg) scale(0.9); filter: blur(0); } 
            90% { transform: translateX(2vw) rotate(0deg) scale(1.05); } 
            100% { transform: translateX(0) rotate(0deg) scale(1); opacity: 1; } 
        }
        @keyframes fadeInText { to { opacity: 1; } }
        @keyframes fadeOut { to { opacity: 0; visibility: hidden; } }

        .main-app { display: flex; flex-direction: column; width: 100%; max-width: 800px; height: 100dvh; background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px); overflow: hidden; position: relative; border-radius: 0; }
        
        @media (min-width: 800px) {
            body { padding: 20px; }
            .main-app { height: 90vh; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.5); box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
        }

        #sidebar { width: 280px; background: rgba(255, 255, 255, 0.98); border-right: 1px solid rgba(0,0,0,0.1); display: flex; flex-direction: column; position: absolute; height: 100%; z-index: 100; transform: translateX(-100%); transition: 0.3s ease; box-shadow: 2px 0 15px rgba(0,0,0,0.1); }
        #sidebar.open { transform: translateX(0); }
        #sidebar-overlay { display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100dvh; background: rgba(0,0,0,0.4); z-index: 99; }
        #sidebar-overlay.active { display: block; }
        
        .sidebar-header { padding: 15px; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; }
        .sidebar-title { display: flex; align-items: center; gap: 10px; font-weight: bold; font-size: 16px; color: #000; }
        .dog-icon { width: 30px; height: 30px; border-radius: 50%; object-fit: cover; }
        
        #chat-list { flex: 1; min-height: 0; overflow-y: auto; -webkit-overflow-scrolling: touch; padding: 10px; }
        .chat-item { padding: 10px; border-radius: 6px; cursor: pointer; font-size: 13px; color: #475569; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .chat-item:hover { background: #e2e8f0; color: #000; }
        
        .header { background: transparent; color: #000000; padding: 15px 20px; font-size: 18px; font-weight: bold; display: flex; align-items: center; justify-content: space-between; border-bottom: none; }
        .header-left { display: flex; align-items: center; gap: 15px; }
        .menu-btn { background: none; border: none; color: #000; font-size: 24px; cursor: pointer; display: flex; align-items: center; padding: 0; }
        .header-right { display: flex; align-items: center; gap: 12px; }
        
        .icon-btn-header { background: none; border: none; font-size: 22px; cursor: pointer; padding: 5px; display: flex; align-items: center; justify-content: center; color: #000; }
        .icon-btn-header.off { opacity: 0.4; }
        
        #chat-box { flex: 1; min-height: 0; overflow-y: auto; -webkit-overflow-scrolling: touch; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
        
        .msg { max-width: 85%; padding: 12px 16px; border-radius: 18px; line-height: 1.5; font-size: 15px; white-space: pre-wrap; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .user-msg { background: #2563eb; color: white; align-self: flex-end; border-bottom-right-radius: 4px; }
        .ai-msg { background: rgba(255, 255, 255, 0.95); color: #000000; align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid rgba(0,0,0,0.1); }
        
        .input-container { background: #1c1c1e; border-radius: 24px; padding: 10px 15px 15px 15px; margin: 15px; display: flex; flex-direction: column; gap: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        
        .input-top-row { display: flex; align-items: center; justify-content: space-between; }
        .input-pills { display: flex; gap: 8px; }
        
        .pill-btn { background: #2c2c2e; color: #fff; border: none; border-radius: 20px; padding: 6px 12px; font-size: 14px; font-weight: 500; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: 0.2s; }
        .pill-btn.active { background: #3b82f6; }
        .pill-btn:active { background: #3a3a3c; }
        
        .input-right-icons { display: flex; gap: 10px; }
        .icon-btn-input { background: none; border: none; color: #fff; font-size: 20px; cursor: pointer; padding: 0; display: flex; align-items: center; justify-content: center; }
        
        .model-selector-pill { background: transparent; color: #a1a1aa; border: 1px solid #3f3f46; border-radius: 12px; padding: 4px 8px; font-size: 11px; outline: none; margin-bottom: 5px; }
        
        .input-bottom-row { display: flex; align-items: center; gap: 10px; }
        input[type="text"] { flex: 1; background: transparent; border: none; color: #fff; font-size: 16px; outline: none; padding: 5px 0; }
        input[type="text"]::placeholder { color: #71717a; }
        
        #modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100dvh; background: rgba(0,0,0,0.6); z-index: 1000; opacity: 0; transition: opacity 0.3s ease; }
        #modal-overlay.active { display: block; opacity: 1; }

        #action-modal {
            position: fixed; bottom: 0; left: 0; width: 100%; 
            background: #1c1c1e; 
            border-radius: 24px 24px 0 0;
            z-index: 1001;
            transform: translateY(100%);
            transition: transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1);
            padding: 10px 20px 30px 20px;
            box-sizing: border-box;
            display: flex; flex-direction: column;
        }
        #action-modal.open { transform: translateY(0); }

        .modal-drag-handle { width: 40px; height: 5px; background: #555; border-radius: 10px; margin: 0 auto 20px auto; }
        
        .modal-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        .grid-btn { background: #2c2c2e; border-radius: 16px; padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; border: none; color: white; transition: background 0.2s; }
        .grid-btn:active { background: #3a3a3c; }
        .grid-btn .icon { font-size: 28px; margin-bottom: 8px; }
        .grid-btn span { font-size: 15px; font-weight: 500; }

        .modal-list { display: flex; flex-direction: column; }
        .list-item { display: flex; align-items: center; gap: 15px; padding: 15px 0; border-bottom: 1px solid #333; cursor: pointer; border: none; background: none; text-align: left; color: white; }
        .list-item:last-child { border-bottom: none; }
        .list-item .icon { font-size: 24px; width: 30px; text-align: center; }
        .list-item-text { display: flex; flex-direction: column; }
        .list-item-title { font-size: 16px; font-weight: 600; margin-bottom: 4px; }
        .list-item-sub { font-size: 13px; color: #888; }

        input[type="file"] { opacity: 0; width: 0.1px; height: 0.1px; position: absolute; z-index: -1; }
        
        /* UPDATED: Button container for Listen and Copy */
        .msg-actions { display: flex; gap: 8px; margin-top: 10px; }
        .speak-btn { background: #f1f5f9; border: 1px solid #cbd5e1; color: #1e293b; padding: 6px 10px; border-radius: 8px; font-size: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; transition: 0.2s; }
        .speak-btn:hover { background: #e2e8f0; }
    </style>
</head>
<body>
    <div id="splash-screen">
        <img src="/icon.png" class="splash-logo" alt="Churky">
        <div class="splash-text">CHURKY INITIALIZING...</div>
    </div>

    <div id="sidebar-overlay" onclick="toggleSidebar()"></div>
    <div id="modal-overlay" onclick="closeModal()"></div>

    <div class="main-app" id="main-app">
        <div id="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-title">
                    <img src="/icon.png" class="dog-icon" alt="Churky"> CHURKY PAGES
                </div>
                <button onclick="toggleSidebar()" style="background: none; color: #000; padding: 5px; font-size: 18px;">✕</button>
            </div>
            <div id="chat-list"></div>
        </div>
        
        <div class="header">
            <div class="header-left">
                <button class="menu-btn" onclick="toggleSidebar()">☰</button>
                CHURKY
            </div>
            <div class="header-right">
                <button id="auto-read-btn" class="icon-btn-header" onclick="toggleAutoRead()">🔊</button>
                <button class="icon-btn-header" onclick="startNewChat()">⊕</button>
            </div>
        </div>
        
        <div id="chat-box">
            <div class="msg ai-msg" id="welcome-msg">Welcome back, Fernando. I am ready.</div>
        </div>

        <div class="input-container">
            <div class="input-top-row">
                <div class="input-pills">
                    <select id="model-selector" class="model-selector-pill">
                        <option value="openrouter/free">Genius</option>
                        <option value="openai/gpt-oss-120b">Fast</option>
                        <option value="ollama/qwen2.5:3b">Private (Local)</option>
                    </select>
                    <button class="pill-btn" id="think-btn" onclick="toggleWebMode()">✨ Think</button>
                    <button class="pill-btn" onclick="triggerTool('Image')">🌐 Search</button>
                </div>
                <div class="input-right-icons">
                    <button class="icon-btn-input" onclick="openModal()">⊕</button>
                    <button class="icon-btn-input" id="mic-btn" onclick="toggleRecording()">🎤</button>
                </div>
            </div>
            
            <div class="input-bottom-row">
                <input type="file" id="img-input" accept="image/*" onchange="handleImage()">
                <input type="file" id="camera-input" accept="image/*" capture="environment" onchange="handleImage()">
                <input type="file" id="doc-input" accept=".pdf,.docx,.txt" onchange="handleDoc()">
                <input type="text" id="msg-input" placeholder="Type a message or hold to speak..." onkeypress="if(event.key === 'Enter') sendMessage()">
            </div>
        </div>
    </div>

    <div id="action-modal">
        <div class="modal-drag-handle"></div>
        
        <div class="modal-grid">
            <button class="grid-btn" onclick="document.getElementById('img-input').click(); closeModal();">
                <span class="icon">🖼️</span><span>Photos</span>
            </button>
            <button class="grid-btn" onclick="document.getElementById('camera-input').click(); closeModal();">
                <span class="icon">📷</span><span>Camera</span>
            </button>
        </div>

        <div class="modal-list">
            <button class="list-item" onclick="triggerTool('Image'); closeModal();">
                <span class="icon">🎨</span>
                <div class="list-item-text">
                    <span class="list-item-title">Create images</span>
                    <span class="list-item-sub">Visualize and edit (Flux)</span>
                </div>
            </button>
            <button class="list-item" onclick="triggerTool('Web'); closeModal();">
                <span class="icon">🌐</span>
                <div class="list-item-text">
                    <span class="list-item-title">Deep research</span>
                    <span class="list-item-sub">Get live web results</span>
                </div>
            </button>
            <button class="list-item" onclick="triggerTool('Code'); closeModal();">
                <span class="icon">💻</span>
                <div class="list-item-text">
                    <span class="list-item-title">Canvas</span>
                    <span class="list-item-sub">Code, write, or make slides</span>
                </div>
            </button>
            <button class="list-item" onclick="triggerTool('Fix'); closeModal();">
                <span class="icon">🔧</span>
                <div class="list-item-text">
                    <span class="list-item-title">Guided learning</span>
                    <span class="list-item-sub">Fix bugs step-by-step</span>
                </div>
            </button>
            <button class="list-item" onclick="document.getElementById('doc-input').click(); closeModal();">
                <span class="icon">📄</span>
                <div class="list-item-text">
                    <span class="list-item-title">More uploads</span>
                    <span class="list-item-sub">Files, Notebooks, and more</span>
                </div>
            </button>
        </div>
    </div>

    <script>
        if ('serviceWorker' in navigator) { navigator.serviceWorker.register('/sw.js'); }
        
        setTimeout(() => {
            const splash = document.getElementById('splash-screen');
            if (splash) splash.remove();
        }, 3000);

        let currentChatId = null, imgBase64 = null, mediaRecorder, audioChunks = [], isRecording = false;
        let availableVoices = [], autoRead = true, audioUnlocked = false; 
        let webMode = false;
        
        const chatBox = document.getElementById('chat-box'), input = document.getElementById('msg-input'), micBtn = document.getElementById('mic-btn'), sidebar = document.getElementById('sidebar'), overlay = document.getElementById('sidebar-overlay'), chatList = document.getElementById('chat-list'), modelSelector = document.getElementById('model-selector'), autoReadBtn = document.getElementById('auto-read-btn'), thinkBtn = document.getElementById('think-btn');
        const actionModal = document.getElementById('action-modal'), modalOverlay = document.getElementById('modal-overlay');

        document.getElementById('main-app').addEventListener('click', () => {
            if (!audioUnlocked) {
                audioUnlocked = true;
                const welcomeText = document.getElementById('welcome-msg').innerText;
                if (autoRead) speakText(welcomeText);
            }
        }, { once: true });

        function openModal() { actionModal.classList.add('open'); modalOverlay.classList.add('active'); }
        function closeModal() { actionModal.classList.remove('open'); modalOverlay.classList.remove('active'); }

        function toggleAutoRead() {
            autoRead = !autoRead;
            if (autoRead) { autoReadBtn.classList.remove('off'); autoReadBtn.innerText = '🔊'; } 
            else { autoReadBtn.classList.add('off'); autoReadBtn.innerText = '🔇'; window.speechSynthesis.cancel(); }
        }

        function toggleWebMode() {
            webMode = !webMode;
            if (webMode) {
                thinkBtn.classList.add('active');
                input.placeholder = "Search the web...";
            } else {
                thinkBtn.classList.remove('active');
                input.placeholder = "Type a message...";
            }
            input.focus();
        }

        function toggleSidebar() { sidebar.classList.toggle('open'); overlay.classList.toggle('active'); if (sidebar.classList.contains('open')) loadChats(); }
        
        function startNewChat() { 
            sidebar.classList.remove('open'); 
            overlay.classList.remove('active'); 
            currentChatId = null; 
            imgBase64 = null; 
            chatBox.innerHTML = '<div class="msg ai-msg" id="welcome-msg">New session started.</div>'; 
            input.value = ''; 
            input.disabled = false;
            setTimeout(() => { input.focus(); }, 100);
            if (autoRead && audioUnlocked) speakText("New session started."); 
        }
        
        function triggerTool(tool) { 
            if (tool === 'Image') { 
                input.value = "/image a highly detailed, photorealistic, 8k resolution photograph of "; 
                input.focus(); 
            } else if (tool === 'Code') { 
                input.value = "Write a complete, working script for the following request: "; 
                input.focus(); 
            } else if (tool === 'Fix') { 
                input.value = "Find the bugs in this code and provide the corrected version:\n\n"; 
                input.focus(); 
            } else if (tool === 'Web') {
                webMode = true;
                thinkBtn.classList.add('active');
                input.placeholder = "Search the web...";
                input.focus();
                addMsg("Web search mode activated for the next message.", "ai-msg", false);
            }
        }
        
        function loadVoices() { 
            availableVoices = window.speechSynthesis.getVoices(); 
            if (availableVoices.length === 0) setTimeout(loadVoices, 100);
        }
        if (typeof window !== 'undefined' && window.speechSynthesis) { 
            window.speechSynthesis.onvoiceschanged = loadVoices; 
            loadVoices(); 
        }

        // UPGRADED VOICE FUNCTION FOR PREMIUM VOICES
        function speakText(text) {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                
                // Search for highest quality male voices first (Premium/Enhanced)
                let maleVoice = availableVoices.find(v => 
                    (v.name.includes('Aaron') || v.name.includes('Daniel') || v.name.includes('Alex')) && 
                    (v.name.includes('Premium') || v.name.includes('Enhanced'))
                );

                // Fallback to standard male voices if premium isn't downloaded
                if (!maleVoice) {
                    maleVoice = availableVoices.find(v => 
                        v.name.includes('Aaron') || v.name.includes('Daniel') || v.name.includes('Alex') || 
                        v.name.includes('David') || v.name.includes('Mark') || v.name.includes('George')
                    );
                }
                
                if (maleVoice) {
                    utterance.voice = maleVoice;
                } else {
                    // If no male voice is found at all, drop the pitch so it sounds male
                    utterance.pitch = 0.7;
                }
                
                utterance.rate = 1.0; 
                window.speechSynthesis.speak(utterance);
            }
        }

        // NEW: COPY TO CLIPBOARD FUNCTION
        function copyText(text, btnElement) {
            navigator.clipboard.writeText(text).then(() => {
                const originalText = btnElement.innerHTML;
                btnElement.innerHTML = '✅ Copied!';
                setTimeout(() => { btnElement.innerHTML = originalText; }, 2000);
            }).catch(err => {
                console.error('Failed to copy: ', err);
            });
        }

        async function loadChats() {
            try {
                const res = await fetch('/chats'); if (!res.ok) throw new Error("Server Error");
                const chats = await res.json(); chatList.innerHTML = '';
                if (!chats.length) chatList.innerHTML = '<div style="text-align:center; color:#94a3b8; font-size:12px; margin-top:20px;">No history yet.</div>';
                chats.forEach(chat => { const div = document.createElement('div'); div.className = 'chat-item'; div.innerText = chat.name; div.onclick = () => loadChat(chat.id); chatList.appendChild(div); });
            } catch(e) { chatList.innerHTML = '<div style="color:red; font-size:12px;">Error: ' + e.message + '</div>'; }
        }
        
        async function loadChat(chatId) {
            currentChatId = chatId; toggleSidebar(); chatBox.innerHTML = '<div class="msg ai-msg">Loading...</div>';
            try {
                const res = await fetch(`/chats/${chatId}`); const history = await res.json(); chatBox.innerHTML = '';
                history.forEach(msg => { if (msg.role === 'user') addMsg(Array.isArray(msg.content) ? msg.content.find(c=>c.type==='text')?.text : msg.content, 'user-msg', false); else if (msg.role === 'assistant') addMsg(msg.content, 'ai-msg', false); });
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch(e) { chatBox.innerHTML = '<div class="msg ai-msg">Failed to load.</div>'; }
        }
        
        function addMsg(text, type, readAloud = true) {
            const div = document.createElement('div'); div.className = `msg ${type}`;
            
            let cleanText = text;
            if (type === 'ai-msg') {
                cleanText = cleanText.replace(/\*\*/g, ''); 
                cleanText = cleanText.replace(/---/g, '');  
                cleanText = cleanText.replace(/__/g, '');   
                cleanText = cleanText.replace(/^#+\s/gm, ''); 
            }
            
            const textNode = document.createElement('span'); textNode.innerText = cleanText; div.appendChild(textNode);
            
            if (type === 'ai-msg') {
                // Create the button container
                const btnContainer = document.createElement('div');
                btnContainer.className = 'msg-actions';

                // Listen Button
                const listenBtn = document.createElement('button'); 
                listenBtn.className = 'speak-btn'; 
                listenBtn.innerHTML = '🔊 Listen';
                listenBtn.onclick = () => { if (window.speechSynthesis.speaking) window.speechSynthesis.cancel(); else speakText(cleanText); }; 
                btnContainer.appendChild(listenBtn);

                // NEW: Copy Button
                const copyBtn = document.createElement('button');
                copyBtn.className = 'speak-btn';
                copyBtn.innerHTML = '📋 Copy';
                copyBtn.onclick = () => copyText(cleanText, copyBtn);
                btnContainer.appendChild(copyBtn);

                div.appendChild(btnContainer);
                
                if (autoRead && readAloud && audioUnlocked) speakText(cleanText);
            }
            chatBox.appendChild(div); chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function toggleRecording() {
            if ('speechSynthesis' in window) window.speechSynthesis.cancel();
            if (isRecording) { mediaRecorder.stop(); isRecording = false; micBtn.style.color = '#fff'; return; }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); mediaRecorder = new MediaRecorder(stream); audioChunks = [];
                mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); const formData = new FormData(); formData.append("audio", audioBlob, "voice.webm");
                    addMsg("Transcribing audio...", "ai-msg", false);
                    try {
                        const res = await fetch('/transcribe', { method: 'POST', body: formData }); const data = await res.json();
                        if (res.ok) { document.querySelectorAll('.ai-msg').forEach(el => { if(el.innerText.includes("Transcribing audio...")) el.remove(); }); input.value = data.text; } 
                    } catch(e) { addMsg("Voice error.", "ai-msg", false); }
                };
                mediaRecorder.start(); isRecording = true; micBtn.style.color = 'red';
            } catch (err) { addMsg("Microphone access denied.", "ai-msg", false); }
        }
        
        function handleImage() { 
            const file = document.getElementById('img-input').files[0] || document.getElementById('camera-input').files[0]; 
            if (!file) return; 
            const reader = new FileReader(); 
            reader.onload = function(e) { 
                imgBase64 = e.target.result.split(',')[1]; 
                addMsg("[Image Attached]", "ai-msg", false); 
            }; 
            reader.readAsDataURL(file); 
        }
        
        async function handleDoc() {
            const file = document.getElementById('doc-input').files[0]; if (!file) return;
            const formData = new FormData(); formData.append("file", file); addMsg("Reading document...", "ai-msg", false);
            try { const res = await fetch('/upload_doc', { method: 'POST', body: formData }); const data = await res.json(); if (res.ok) input.value = "Analyze this document: " + data.text; } catch(e) {}
        }

        async function sendMessage() {
            if ('speechSynthesis' in window) window.speechSynthesis.cancel();
            const text = input.value.trim(); if (!text && !imgBase64) return; input.value = '';
            
            const selectedModel = modelSelector.value;
            const originalText = text; 
            addMsg(text + (imgBase64 ? " [Image]" : ""), 'user-msg', false); 
            input.disabled = true;
            
            const thinkingId = "thinking-" + Date.now();
            const thinkingDiv = document.createElement('div'); thinkingDiv.className = 'msg ai-msg'; thinkingDiv.id = thinkingId; thinkingDiv.innerText = webMode ? 'Searching the web...' : 'Churky is thinking...'; chatBox.appendChild(thinkingDiv); chatBox.scrollTop = chatBox.scrollHeight;

            try {
                let finalMessage = text;

                if (webMode && text) {
                    try {
                        const searchRes = await fetch('/web_search?query=' + encodeURIComponent(text));
                        const searchData = await searchRes.json();
                        finalMessage = `[WEB SEARCH RESULTS]:\n${searchData.context}\n\n[USER QUESTION]: ${text}\n\nInstructions: Answer the user's question accurately using the web search results above. Do not mention the search results, just give a natural answer.`;
                        document.getElementById(thinkingId).innerText = 'Churky is reading results...';
                    } catch(e) { console.error("Search failed", e); }
                    webMode = false;
                    thinkBtn.classList.remove('active');
                    input.placeholder = "Type a message...";
                }

                if (text.toLowerCase().startsWith("/image") || text.toLowerCase().includes("generate an image")) {
                    let promptText = text.replace(/\/image/gi, '').replace(/generate an image/gi, '').trim();
                    const res = await fetch('/edit_image?prompt=' + encodeURIComponent(promptText), { method: 'POST' });
                    const data = await res.json(); document.getElementById(thinkingId).remove();
                    if (res.ok) {
                        const imgDiv = document.createElement('div'); imgDiv.className = 'msg ai-msg'; imgDiv.innerHTML = `<p>Here is the realistic image:</p><img src="${data.url}" style="width:100%; border-radius:10px; margin-top:10px;">`; chatBox.appendChild(imgDiv); chatBox.scrollTop = chatBox.scrollHeight;
                        if (autoRead && audioUnlocked) speakText("Here is your image.");
                    } else addMsg("Error: " + data.detail, 'ai-msg', false);
                } else {
                    const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: finalMessage, original_message: originalText, chat_id: currentChatId, image_base64: imgBase64, model: selectedModel }) });
                    if (!res.ok) { const errorData = await res.json(); throw new Error(errorData.detail || "Server Error"); }
                    const data = await res.json(); document.getElementById(thinkingId).remove(); addMsg(data.reply, 'ai-msg', true); if(data.chat_id) currentChatId = data.chat_id; 
                }
            } catch (e) { 
                const thinkEl = document.getElementById(thinkingId); if (thinkEl) thinkEl.remove(); 
                addMsg("CRASH LOG: " + e.message, 'ai-msg', false); 
            }
            imgBase64 = null; document.getElementById('img-input').value = ''; document.getElementById('camera-input').value = ''; input.disabled = false; input.focus();
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    print("Starting Churky Co-Pilot...")
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)