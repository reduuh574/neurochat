import streamlit as st
import os
import json
import time
from datetime import datetime
from pathlib import Path

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="NeuroChat — Assistant IA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ──────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ddgs import DDGS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CHROMA_DIR  = BASE_DIR / "chroma_db"
MEMORY_FILE = BASE_DIR / "memory.json"
UPLOADS_DIR = BASE_DIR / "uploads"
CHROMA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3d;
    --accent: #6c63ff;
    --accent2: #ff6584;
    --text: #e8e8f0;
    --text-muted: #7a7a9a;
    --user-bubble: #1e1e35;
    --ai-bubble: #161624;
    --success: #4ade80;
    --warning: #fbbf24;
    --audio: #f472b6;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.main .block-container {
    padding: 1.5rem 2rem 6rem 2rem;
    max-width: 900px;
}

.app-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    letter-spacing: -0.5px;
}
.app-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.msg-row {
    display: flex;
    margin-bottom: 1.2rem;
    gap: 0.75rem;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.avatar.ai   { background: linear-gradient(135deg, var(--accent), #9b59b6); }
.avatar.user { background: linear-gradient(135deg, var(--accent2), #ff8c42); }

.bubble {
    max-width: 75%;
    padding: 0.85rem 1.1rem;
    border-radius: 16px;
    font-size: 0.93rem;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}
.bubble.ai {
    background: var(--ai-bubble);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
}
.bubble.user {
    background: var(--user-bubble);
    border: 1px solid #2d2d4d;
    border-top-right-radius: 4px;
    text-align: right;
}

.source-tag {
    display: inline-block;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 0.5rem;
    margin-right: 4px;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.source-pdf   { background: #1a2a1a; color: var(--success);  border: 1px solid #2a4a2a; }
.source-web   { background: #2a1a1a; color: var(--warning);  border: 1px solid #4a2a1a; }
.source-audio { background: #2a1a2a; color: var(--audio);    border: 1px solid #4a1a4a; }
.source-llm   { background: #1a1a2a; color: var(--accent);   border: 1px solid #2a2a4a; }

.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9b59b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; }

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("vectorstore", None),
    ("embeddings", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_memory():
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(messages: list):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def get_embeddings():
    if st.session_state.embeddings is None:
        with st.spinner("Chargement du modèle d'embeddings…"):
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
    return st.session_state.embeddings

def get_vectorstore():
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name="knowledge_base",
    )

def count_vectorstore_docs() -> int:
    try:
        return get_vectorstore()._collection.count()
    except:
        return 0

def ingest_pdf(uploaded_file) -> int:
    """Save PDF, chunk it, add to ChromaDB. Returns chunk count."""
    save_path = UPLOADS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(str(save_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    get_vectorstore().add_documents(chunks)
    return len(chunks)

def ingest_audio(uploaded_file) -> tuple[int, str]:
    """
    Transcribe audio with Whisper, chunk text, add to ChromaDB.
    Returns (chunk_count, transcription_preview).
    """
    import whisper

    save_path = UPLOADS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    model = whisper.load_model("base")
    result = model.transcribe(str(save_path))
    text = result["text"]

    if not text.strip():
        return 0, ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": uploaded_file.name, "type": "audio"}]
    )
    get_vectorstore().add_documents(chunks)
    return len(chunks), text

def search_web(query: str, max_results: int = 4) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        return "\n\n---\n\n".join(
            f"**{r.get('title','')}**\n{r.get('body','')}\nSource: {r.get('href','')}"
            for r in results
        )
    except Exception as e:
        return f"Erreur de recherche web : {e}"

def get_llm(api_key: str):
    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=1024,
    )

def build_history(messages: list, max_turns: int = 8):
    lc = []
    for m in messages[-(max_turns * 2):]:
        if m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        else:
            lc.append(AIMessage(content=m["content"]))
    return lc

def answer_question(query: str, api_key: str, search_mode: str) -> dict:
    """
    Route: PDF/Audio RAG → Web Search → Pure LLM.
    Returns {"answer": str, "source": str}
    """
    llm = get_llm(api_key)
    source = "llm"
    context_parts = []

    # 1 ── RAG (PDFs + audios in ChromaDB)
    if search_mode in ("Documents + Web", "Documents seulement"):
        try:
            vs = get_vectorstore()
            if vs._collection.count() > 0:
                docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(query)
                if docs:
                    context_parts.append(
                        "[BASE DOCUMENTAIRE]\n" +
                        "\n\n".join(d.page_content for d in docs)
                    )
                    source = "pdf"
        except Exception as e:
            st.warning(f"Erreur RAG : {e}")

    # 2 ── Web search
    if search_mode in ("Documents + Web", "Web seulement"):
        if search_mode == "Web seulement" or search_mode == "Documents + Web":
            web = search_web(query)
            if web:
                context_parts.append(f"[RÉSULTATS WEB]\n{web}")
                source = "pdf+web" if source == "pdf" else "web"

    # 3 ── Build prompt + call LLM
    history = build_history(st.session_state.messages)

    if context_parts:
       system = (
        "Tu es NeuroChat, un assistant IA intelligent, précis et bienveillant.\n"
        "You respond in the same language as the user. If they write in French, respond in French. If they write in English, respond in English. Never switch languages mid-response.\n "
        "Si la question est en français, réponds en français."
        "Si la question est en English, respond in English.\n" 
        "RÈGLE ABSOLUE : Tu as accès à des informations en temps réel ci-dessous. "
        "Tu DOIS utiliser ces informations pour répondre. "
        "Ne dis JAMAIS que tu ne connais pas les événements récents ou de 2025. "
        "Les données ci-dessous sont actuelles et fiables — utilise-les directement.\n\n"
        "CONTEXTE DISPONIBLE:\n" + "\n\n".join(context_parts) + "\n\n"
        "Instructions :\n"
        "- Réponds en français uniquement\n"
        "- Base ta réponse sur le CONTEXTE ci-dessus en priorité absolue\n"
        "- Synthétise les informations du contexte de façon claire\n"
        "- Mentionne les sources trouvées\n"
        "- Ne répète JAMAIS que tes connaissances sont limitées à 2024"
    )
    else:
        system = (
            "Tu es NeuroChat, un assistant IA intelligent, précis et bienveillant.\n"
            "Tu réponds TOUJOURS en français, de manière claire et structurée.\n"
            "Instructions :\n"
            "- Réponds en français uniquement\n"
            "- Sois précis, concis et utile\n"
            "- Si tu n'es pas sûr, dis-le clairement"
        )

    msgs = [SystemMessage(content=system)] + history + [HumanMessage(content=query)]
    response = llm.invoke(msgs)
    return {"answer": response.content, "source": source}


# ── Load persisted memory on startup ─────────────────────────────────────────
if not st.session_state.messages:
    stored = load_memory()
    if stored:
        st.session_state.messages = stored


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="font-family:Syne,sans-serif;font-weight:800;font-size:1.3rem;'
        'background:linear-gradient(135deg,#6c63ff,#ff6584);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">'
        '🧠 NeuroChat</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.75rem;color:#7a7a9a;text-transform:uppercase;letter-spacing:0.1em">'
        'Assistant IA Multimodal · Projet 4A</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── API Key
    st.markdown("**🔑 Clé API Groq**")
    api_key = st.text_input(
        "", type="password", placeholder="gsk_...",
        label_visibility="collapsed",
        help="Obtenez votre clé sur console.groq.com",
    )

    st.divider()

    # ── Search mode
    st.markdown("**⚙️ Mode de recherche**")
    search_mode = st.radio(
        "",
        ["Documents + Web", "Documents seulement", "Web seulement", "LLM uniquement"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── PDF Upload
    st.markdown("**📄 Ajouter un document PDF**")
    uploaded_pdf = st.file_uploader(
        "pdf_upload", type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_pdf:
        if (UPLOADS_DIR / uploaded_pdf.name).exists():
            st.info(f"✅ '{uploaded_pdf.name}' déjà indexé")
        else:
            with st.spinner(f"Indexation de {uploaded_pdf.name}…"):
                n = ingest_pdf(uploaded_pdf)
            st.success(f"✅ {n} segments indexés")

    st.divider()

    # ── Audio Upload (Whisper)
    st.markdown("**🎙️ Ajouter un audio (Whisper)**")
    st.caption("MP3, WAV — transcription automatique par IA")
    uploaded_audio = st.file_uploader(
        "audio_upload", type=["mp3", "wav", "m4a"],
        label_visibility="collapsed",
    )
    if uploaded_audio:
        if (UPLOADS_DIR / uploaded_audio.name).exists():
            st.info(f"✅ '{uploaded_audio.name}' déjà transcrit")
        else:
            with st.spinner(f"Transcription de {uploaded_audio.name} avec Whisper…"):
                try:
                    n, transcript = ingest_audio(uploaded_audio)
                    st.success(f"✅ {n} segments indexés")
                    if transcript:
                        with st.expander("📝 Voir la transcription"):
                            st.write(transcript[:1000] + ("…" if len(transcript) > 1000 else ""))
                except ImportError:
                    st.error("⚠️ Installez openai-whisper : pip install openai-whisper")
                except Exception as e:
                    st.error(f"Erreur audio : {e}")

    st.divider()

    # ── Stats
    n_docs = count_vectorstore_docs()
    col_a, col_b = st.columns(2)
    col_a.metric("Segments en base", n_docs)
    col_b.metric("Messages", len(st.session_state.messages))

    # List indexed files
    all_files = list(UPLOADS_DIR.glob("*"))
    if all_files:
        st.markdown("**Fichiers indexés :**")
        for f in all_files:
            icon = "📎" if f.suffix == ".pdf" else "🎙️"
            st.markdown(
                f'<span style="font-size:0.78rem;color:#7a7a9a">{icon} {f.name}</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Memory controls
    st.markdown("**💾 Mémoire**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.session_state.messages = []
            save_memory([])
            st.rerun()
    with col2:
        if st.button("💾 Sauver", use_container_width=True):
            save_memory(st.session_state.messages)
            st.success("Sauvegardé !")

    st.divider()
    st.markdown(
        '<p style="font-size:0.7rem;color:#7a7a9a;text-align:center">'
        'Projet IA · 4ème année · 2025</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="app-title">NeuroChat</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Assistant IA · RAG · Audio Whisper · Recherche Web · Mémoire Persistante</p>',
    unsafe_allow_html=True,
)

SOURCE_LABELS = {
    "pdf":     '<span class="source-tag source-pdf">📄 Base documentaire</span>',
    "web":     '<span class="source-tag source-web">🌐 Recherche web</span>',
    "pdf+web": '<span class="source-tag source-pdf">📄 Docs</span><span class="source-tag source-web">🌐 Web</span>',
    "audio":   '<span class="source-tag source-audio">🎙️ Audio transcrit</span>',
    "llm":     '<span class="source-tag source-llm">🧠 Connaissances LLM</span>',
}

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#7a7a9a">
        <div style="font-size:3rem;margin-bottom:1rem">🧠</div>
        <p style="font-family:Syne,sans-serif;font-size:1.1rem;color:#e8e8f0">Bonjour ! Je suis NeuroChat.</p>
        <p style="font-size:0.85rem">
            Posez-moi n'importe quelle question — je consulte vos documents,<br>
            transcris vos audios, recherche sur le web, et mémorise nos échanges.
        </p>
        <div style="margin-top:1.5rem;display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap">
            <span style="background:#1a1a2a;border:1px solid #2a2a4a;padding:6px 14px;border-radius:20px;font-size:0.78rem">💡 Explique le deep learning</span>
            <span style="background:#1a1a2a;border:1px solid #2a2a4a;padding:6px 14px;border-radius:20px;font-size:0.78rem">📄 Résume mon document</span>
            <span style="background:#1a1a2a;border:1px solid #2a2a4a;padding:6px 14px;border-radius:20px;font-size:0.78rem">🌐 Actualités IA 2025</span>
            <span style="background:#1a1a2a;border:1px solid #2a2a4a;padding:6px 14px;border-radius:20px;font-size:0.78rem">🎙️ Résume ma réunion</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        source  = msg.get("source", "llm")

        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar user">👤</div>
                <div class="bubble user">{content}</div>
            </div>""", unsafe_allow_html=True)
        else:
            badge = SOURCE_LABELS.get(source, SOURCE_LABELS["llm"])
            st.markdown(f"""
            <div class="msg-row ai">
                <div class="avatar ai">🧠</div>
                <div class="bubble ai">{content}<br>{badge}</div>
            </div>""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "", placeholder="Ask your question / Posez votre question...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Envoyer →", use_container_width=True)

if submitted and user_input.strip():
    if not api_key:
        st.error("⚠️ Veuillez entrer votre clé API Groq dans la barre latérale.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input.strip(),
            "timestamp": datetime.now().isoformat(),
        })

        with st.spinner("NeuroChat réfléchit…"):
            try:
                result = answer_question(user_input.strip(), api_key, search_mode)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "source": result["source"],
                    "timestamp": datetime.now().isoformat(),
                })
                save_memory(st.session_state.messages)
                st.rerun()  # ✅ only rerun on success
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.session_state.messages.pop()  # ✅ remove the user message if AI failed