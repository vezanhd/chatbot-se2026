import os
import json
import time
from datetime import datetime
from functools import lru_cache, wraps
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIConnectionError, APIStatusError
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Load environment variables ────────────────────────────────
load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Configuration ─────────────────────────────────────────────
PERSIST_DIR = "./chroma_db"
PDFS_DIR = "pdfs/"
UNANSWERED_LOG = "unanswered_log.json"

# Model configuration - DENGAN FALLBACK STRATEGY
MODEL_FAST = "llama-3.1-8b-instant"           # Untuk query expansion (cepat, kuota besar)
MODEL_SMART = "llama-3.3-70b-versatile"       # Primary: generate jawaban (akurat, kuota kecil)
MODEL_FALLBACK = "llama-3.1-8b-instant"         # Fallback: jika 70B rate limit (kuota besar, cukup akurat)
# Alternatif fallback: "llama-3.1-8b-instant" atau "gemma2-9b-it"

# Token limits (Groq Free Tier)
TOKEN_LIMITS = {
    "llama-3.3-70b-versatile": 100_000,       # 100K tokens/hari
    "llama-3.1-8b-instant": 500_000,          # 500K tokens/hari
}

# Conversation history settings
MAX_HISTORY_LENGTH = 6  # Maksimal 6 pesan terakhir (3 pasang user-assistant)
HISTORY = defaultdict(list)  # session_id -> list of messages

# ── Token Usage Tracker ───────────────────────────────────────
class TokenTracker:
    """Track penggunaan token harian per model untuk antisipasi rate limit"""
    
    def __init__(self):
        self.usage_file = "token_usage.json"
        self.usage = self._load_usage()
    
    def _load_usage(self):
        """Load usage dari file atau reset jika hari baru"""
        today = datetime.now().date().isoformat()
        
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("date") == today:
                        return data
            except (json.JSONDecodeError, IOError):
                pass
        
        # Reset untuk hari baru
        return {
            "date": today,
            "models": defaultdict(lambda: {"tokens": 0, "requests": 0}),
            "total_tokens": 0,
            "total_requests": 0
        }
    
    def _save_usage(self):
        """Simpan usage ke file"""
        try:
            with open(self.usage_file, "w", encoding="utf-8") as f:
                json.dump(self.usage, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[WARNING] Gagal menyimpan token usage: {e}")
    
    def record(self, model: str, tokens_used: int):
        """Catat penggunaan token"""
        self.usage["models"][model]["tokens"] += tokens_used
        self.usage["models"][model]["requests"] += 1
        self.usage["total_tokens"] += tokens_used
        self.usage["total_requests"] += 1
        self._save_usage()
        
        # Warning jika sudah 80% limit
        limit = TOKEN_LIMITS.get(model, 100_000)
        used = self.usage["models"][model]["tokens"]
        percentage = (used / limit) * 100
        
        if percentage >= 90:
            print(f"[CRITICAL] Model {model} sudah {percentage:.1f}% limit! ({used}/{limit} tokens)")
        elif percentage >= 80:
            print(f"[WARNING] Model {model} sudah {percentage:.1f}% limit ({used}/{limit} tokens)")
    
    def get_remaining(self, model: str) -> int:
        """Sisa token untuk model tertentu"""
        limit = TOKEN_LIMITS.get(model, 100_000)
        used = self.usage["models"][model]["tokens"]
        return max(0, limit - used)
    
    def should_fallback(self, model: str, estimated_tokens: int = 2000) -> bool:
        """Cek apakah perlu fallback ke model lain"""
        remaining = self.get_remaining(model)
        # Fallback jika sisa < 5000 tokens atau < estimated_tokens * 2
        return remaining < max(5000, estimated_tokens * 2)
    
    def get_stats(self):
        """Dapatkan statistik penggunaan"""
        stats = {
            "date": self.usage["date"],
            "total_tokens": self.usage["total_tokens"],
            "total_requests": self.usage["total_requests"],
            "models": {}
        }
        
        for model, data in self.usage["models"].items():
            limit = TOKEN_LIMITS.get(model, 100_000)
            stats["models"][model] = {
                "tokens_used": data["tokens"],
                "requests": data["requests"],
                "limit": limit,
                "remaining": max(0, limit - data["tokens"]),
                "percentage": (data["tokens"] / limit) * 100
            }
        
        return stats

token_tracker = TokenTracker()

# ── Logging unanswered questions ──────────────────────────────
def log_unanswered(question: str):
    """Catat pertanyaan yang tidak bisa dijawab untuk iterasi knowledge base"""
    if os.path.exists(UNANSWERED_LOG):
        try:
            with open(UNANSWERED_LOG, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            logs = []
    else:
        logs = []
    
    logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question
    })
    
    with open(UNANSWERED_LOG, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

# ── Retry Decorator ───────────────────────────────────────────
def retry_on_failure(max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    """
    Decorator untuk retry otomatis saat terjadi transient error.
    Menggunakan exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (APIConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        print(f"[RETRY] Attempt {attempt + 1}/{max_retries} gagal: {type(e).__name__}. Retry dalam {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        print(f"[ERROR] Semua {max_retries} attempt gagal: {type(e).__name__}")
                except RateLimitError:
                    # Jangan retry rate limit, langsung raise
                    raise
                except APIStatusError as e:
                    # Retry hanya untuk server errors (5xx)
                    if 500 <= e.status_code < 600 and attempt < max_retries - 1:
                        print(f"[RETRY] Server error {e.status_code}. Retry dalam {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise
            
            raise last_exception
        return wrapper
    return decorator

# ── Load and build knowledge base ─────────────────────────────
def load_knowledge_base():
    """Load dokumen, split, dan buat/load vector store"""
    print("=" * 60)
    print("MEMUAT KNOWLEDGE BASE...")
    print("=" * 60)
    
    # Load dokumen markdown
    print(f"[1/5] Loading dokumen dari {PDFS_DIR}...")
    loader = DirectoryLoader(
        PDFS_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"      ✓ Loaded {len(documents)} dokumen")
    
    if len(documents) == 0:
        print(f"[WARNING] Tidak ada dokumen ditemukan di {PDFS_DIR}")
        print(f"          Pastikan file .md sudah ada di folder tersebut")
    
    # Tambahkan metadata source yang lebih kaya
    print("[2/5] Menambahkan metadata source...")
    for i, doc in enumerate(documents):
        source_path = doc.metadata.get("source", "unknown")
        filename = os.path.basename(source_path)
        
        doc.metadata["source"] = source_path
        doc.metadata["filename"] = filename
        doc.metadata["doc_index"] = i
        
        # Deteksi tipe konten berdasarkan pola
        content = doc.page_content
        if "|" in content and content.count("\n|") > 3:
            doc.metadata["content_type"] = "table"
        elif content.strip().startswith(("- ", "* ", "1. ")):
            doc.metadata["content_type"] = "list"
        else:
            doc.metadata["content_type"] = "text"
    
    # Split dokumen dengan chunking yang lebih baik untuk tabel
    print("[3/5] Splitting dokumen (chunk_size=1500, overlap=300)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=[
            "\n## ",           # Heading 2 - prioritas tertinggi
            "\n### ",          # Heading 3
            "\n#### ",         # Heading 4
            "\n\n",            # Paragraf baru
            "\n|",             # Tabel markdown (PENTING untuk knowledge base SE2026!)
            "\n- ",            # List item
            "\n* ",            # List item alternatif
            "\n1. ",           # Numbered list
            "\n",              # Newline
            ". ",              # Kalimat
            " ",               # Kata
            ""                 # Karakter (fallback)
        ],
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"      ✓ Created {len(chunks)} chunks")
    
    # Tambah chunk index ke metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    # Initialize embeddings
    print("[4/5] Initializing embeddings (multilingual-MiniLM-L12-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # Improve similarity search
    )
    
    # Load atau buat vector store
    print("[5/5] Loading/membuat vector store...")
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print(f"      ✓ Loading existing vector store dari {PERSIST_DIR}...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        print(f"      ✓ Loaded {vectorstore._collection.count()} vectors")
    else:
        print(f"      ⚡ Building vector store baru (ini akan memakan waktu)...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        vectorstore.persist()
        print(f"      ✓ Vector store tersimpan di {PERSIST_DIR}")
    
    print("=" * 60)
    print("KNOWLEDGE BASE SIAP!")
    print(f"Total chunks: {len(chunks)}")
    print("=" * 60)
    return vectorstore, embeddings

# Load knowledge base saat aplikasi start
print("\n🚀 Starting SE2026 Chatbot...\n")
vectorstore, embeddings = load_knowledge_base()

# ── Query expansion dengan caching ────────────────────────────
@lru_cache(maxsize=200)
def expand_query(query: str) -> str:
    """
    Ubah pertanyaan informal ke kata kunci formal SE2026.
    Hasil di-cache untuk efisiensi (max 200 query unik).
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[
                {
                    "role": "system",
                    "content": """Kamu membantu mencari informasi di modul Sensus Ekonomi 2026 (SE2026) dari BPS Indonesia.
Ubah pertanyaan berikut menjadi beberapa kata kunci formal yang relevan dengan SE2026.

Contoh:
- "masjid di data ga?" → "tempat ibadah, kode penggunaan bangunan, pendataan SE2026"
- "usaha keliling dicatat dimana?" → "usaha keliling, pencatatan, bangunan tempat tinggal"
- "stiker dipasang kapan?" → "pemasangan stiker, nomor urut bangunan, pendataan lapangan"
- "gorengan keliling kbli apa?" → "gorengan, KBLI, kategori C, kategori I, penyediaan makan minum"

Jawab HANYA dengan kata kunci, pisahkan dengan koma. Maksimal 10 kata kunci."""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0,
            max_tokens=100
        )
        
        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            token_tracker.record(MODEL_FAST, response.usage.total_tokens)
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Query expansion gagal: {e}")
        return query

# ── Smart retrieval dengan diversity ──────────────────────────
def retrieve_context(query: str, k: int = 6) -> tuple[list, str]:
    """
    Retrieve chunks dengan diversity (jangan semua dari dokumen yang sama).
    Return: (relevant_docs, context_string)
    """
    # Ambil lebih banyak kandidat
    candidates = vectorstore.similarity_search_with_score(query, k=k * 2)
    
    # Diversifikasi: max 2 chunks per dokumen
    selected = []
    doc_counts = defaultdict(int)
    
    for doc, score in candidates:
        filename = doc.metadata.get("filename", "unknown")
        if doc_counts[filename] < 2:  # Max 2 chunks per file
            selected.append(doc)
            doc_counts[filename] += 1
            
            if len(selected) >= k:
                break
    
    # Fallback jika tidak cukup diverse
    if len(selected) < k:
        for doc, score in candidates:
            if doc not in selected:
                selected.append(doc)
                if len(selected) >= k:
                    break
    
    context = "\n\n---\n\n".join([doc.page_content for doc in selected])
    return selected, context

# ── Chat completion dengan fallback ───────────────────────────
@retry_on_failure(max_retries=2, initial_delay=1.0)
def chat_completion_with_fallback(messages: list, temperature: float = 0.3, max_tokens: int = 1500) -> str:
    """
    Chat completion dengan fallback strategy:
    1. Coba MODEL_SMART (70B) - paling akurat
    2. Jika rate limit, fallback ke MODEL_FALLBACK (Mixtral 8x7B)
    3. Jika masih gagal, fallback ke MODEL_FAST (8B)
    """
    
    # Cek apakah perlu langsung fallback
    if token_tracker.should_fallback(MODEL_SMART, max_tokens):
        print(f"[FALLBACK] Token {MODEL_SMART} hampir habis, langsung pakai {MODEL_FALLBACK}")
        primary_model = MODEL_FALLBACK
        secondary_model = MODEL_FAST
    else:
        primary_model = MODEL_SMART
        secondary_model = MODEL_FALLBACK
    
    # Coba primary model
    try:
        response = client.chat.completions.create(
            model=primary_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            token_tracker.record(primary_model, response.usage.total_tokens)
        
        print(f"[MODEL] Menggunakan: {primary_model}")
        return response.choices[0].message.content.strip()
    
    except RateLimitError as e:
        print(f"[RATE LIMIT] {primary_model} rate limited: {str(e)[:100]}...")
        print(f"[FALLBACK] Beralih ke {secondary_model}...")
        
        # Fallback ke secondary model
        try:
            response = client.chat.completions.create(
                model=secondary_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if hasattr(response, 'usage') and response.usage:
                token_tracker.record(secondary_model, response.usage.total_tokens)
            
            print(f"[MODEL] Fallback berhasil: {secondary_model}")
            return response.choices[0].message.content.strip()
        
        except RateLimitError:
            # Semua model rate limited
            print(f"[CRITICAL] Semua model rate limited!")
            return "Maaf, sistem sedang mengalami beban tinggi. Silakan coba lagi dalam beberapa menit atau hubungi admin untuk upgrade paket API."
        
        except Exception as fallback_error:
            print(f"[ERROR] Fallback juga gagal: {type(fallback_error).__name__}: {fallback_error}")
            raise
    
    except APIStatusError as e:
        if e.status_code == 429:  # Rate limit juga bisa muncul sebagai 429
            print(f"[RATE LIMIT 429] {primary_model} rate limited")
            raise RateLimitError(str(e))
        raise

# ── Response post-processing ──────────────────────────────────
def post_process_response(response_text: str) -> str:
    """
    Post-process response untuk memastikan format terjaga:
    - Tabel markdown tidak ter-break
    - Formatting konsisten
    """
    # Pastikan tidak ada trailing whitespace berlebihan
    response_text = response_text.strip()
    
    # Fix tabel markdown yang ter-break (baris tanpa | di tengah tabel)
    lines = response_text.split('\n')
    fixed_lines = []
    in_table = False
    
    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|'):
            in_table = True
            fixed_lines.append(line)
        elif in_table and '|' in line:
            fixed_lines.append(line)
        elif in_table and not line.strip():
            # Empty line bisa mengakhiri tabel
            in_table = False
            fixed_lines.append(line)
        else:
            if in_table and not line.strip().startswith('|'):
                in_table = False
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

# ── System prompt untuk generate jawaban ──────────────────────
SYSTEM_PROMPT = """Kamu adalah **Asisten Virtual Ahli Sensus Ekonomi 2026 (SE2026)** dari Badan Pusat Statistik (BPS) Indonesia.

TUGAS UTAMA:
Jawab pertanyaan petugas lapangan (PPL/PML) dan responden SE2026 berdasarkan KONTEKS DOKUMEN yang diberikan.

ATURAN PENTING:
1. Jawab HANYA berdasarkan konteks dokumen SE2026 yang diberikan di bawah
2. Jika informasi TIDAK ADA di konteks, jawab dengan kalimat:
   "Maaf, informasi tersebut tidak saya temukan dalam dokumen SE2026. Silakan modifikasi pertanyaan Anda atau hubungi pusat bantuan SE2026 di halose2026@bps.go.id / WhatsApp 0815-1126-2026."
3. JANGAN mengarang informasi dari pengetahuan umum di luar dokumen
4. Jawab dalam Bahasa Indonesia yang jelas, ringkas, dan mudah dipahami petugas lapangan
5. Jika relevan, sebutkan nomor rincian, blok kuesioner, atau kode KBLI yang terkait
6. Gunakan format yang terstruktur: bullet points, numbering, atau tabel jika perlu
7. Untuk kasus batas (seperti penentuan KBLI, jumlah establishment, dll), jelaskan alur keputusannya secara bertahap
8. Hindari pengulangan informasi yang sama

GAYA JAWABAN:
- Profesional namun ramah
- Gunakan istilah teknis SE2026 dengan tepat (KBLI, BKU, SLS, establishment, dll)
- Berikan contoh konkret jika memungkinkan
- Jika pertanyaan ambigu, berikan klarifikasi dulu sebelum menjawab

KONTEKS DOKUMEN SE2026:
{context}"""

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve halaman utama chatbot"""
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint dengan info token usage"""
    token_stats = token_tracker.get_stats()
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vectorstore_size": vectorstore._collection.count() if hasattr(vectorstore, '_collection') else 0,
        "active_sessions": len(HISTORY),
        "token_usage": token_stats,
        "models": {
            "fast": MODEL_FAST,
            "smart": MODEL_SMART,
            "fallback": MODEL_FALLBACK
        }
    })

@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint utama chatbot.
    Request body: {"message": "...", "session_id": "..."} (session_id optional)
    """
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Field 'message' is required"}), 400
        
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        session_id = data.get("session_id", "default")
        
        print(f"\n{'='*60}")
        print(f"[CHAT] Session: {session_id}")
        print(f"[USER] {user_message}")
        
        # ── Step 1: Query Expansion (dengan cache) ────────────
        expanded_query = expand_query(user_message)
        search_query = f"{user_message} {expanded_query}"
        print(f"[EXPANDED] {expanded_query}")
        
        # ── Step 2: Retrieve relevant chunks (dengan diversity) ─
        relevant_docs, context = retrieve_context(search_query, k=6)
        
        # Debug chunks
        if os.getenv("DEBUG", "false").lower() == "true":
            print(f"\n[CHUNKS RETRIEVED] {len(relevant_docs)} chunks")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"  Chunk {i} ({doc.metadata.get('filename', 'unknown')}, type: {doc.metadata.get('content_type', 'text')}):")
                print(f"    {doc.page_content[:150]}...")
        
        # ── Step 3: Build conversation history ────────────────
        history_messages = HISTORY[session_id][-MAX_HISTORY_LENGTH:]
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}
        ]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_message})
        
        # ── Step 4: Generate response dengan fallback ─────────
        response_text = chat_completion_with_fallback(
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        # ── Step 5: Post-process response ─────────────────────
        response_text = post_process_response(response_text)
        print(f"[BOT] {response_text[:200]}...")
        
        # ── Step 6: Save to conversation history ──────────────
        HISTORY[session_id].append({"role": "user", "content": user_message})
        HISTORY[session_id].append({"role": "assistant", "content": response_text})
        
        # Batasi ukuran history agar tidak terlalu panjang
        if len(HISTORY[session_id]) > MAX_HISTORY_LENGTH * 2:
            HISTORY[session_id] = HISTORY[session_id][-MAX_HISTORY_LENGTH * 2:]
        
        # ── Step 7: Log unanswered questions ──────────────────
        if "tidak saya temukan" in response_text.lower():
            log_unanswered(user_message)
            print(f"[UNANSWERED] Logged: {user_message}")
        
        return jsonify({
            "response": response_text,
            "session_id": session_id,
            "model_used": "auto"  # Bisa ditambahkan tracking model yang dipakai
        })
    
    except RateLimitError as e:
        print(f"[CRITICAL] Semua model rate limited: {e}")
        return jsonify({
            "response": "Maaf, sistem sedang mengalami beban tinggi karena banyak permintaan. Silakan coba lagi dalam 5-10 menit. Jika masalah berlanjut, hubungi admin untuk upgrade paket API Groq.",
            "error": "rate_limit_exceeded"
        }), 503
    
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return jsonify({
            "response": "Maaf, terjadi kesalahan teknis saat memproses permintaan Anda. Silakan coba beberapa saat lagi.",
            "error": str(e) if os.getenv("DEBUG", "false").lower() == "true" else None
        }), 500

@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Bersihkan conversation history untuk session tertentu"""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    
    if session_id in HISTORY:
        del HISTORY[session_id]
        return jsonify({"status": "success", "message": f"History untuk session '{session_id}' telah dihapus"})
    
    return jsonify({"status": "success", "message": "Session tidak ditemukan"})

@app.route("/unanswered", methods=["GET"])
def get_unanswered():
    """Lihat semua pertanyaan yang tidak terjawab"""
    if not os.path.exists(UNANSWERED_LOG):
        return jsonify({"total": 0, "questions": []})
    
    try:
        with open(UNANSWERED_LOG, "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        limit = request.args.get("limit", default=50, type=int)
        
        return jsonify({
            "total": len(logs),
            "questions": logs[-limit:]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Statistik penggunaan chatbot yang komprehensif"""
    unanswered_count = 0
    if os.path.exists(UNANSWERED_LOG):
        try:
            with open(UNANSWERED_LOG, "r", encoding="utf-8") as f:
                unanswered_count = len(json.load(f))
        except:
            pass
    
    token_stats = token_tracker.get_stats()
    
    return jsonify({
        "vectorstore_size": vectorstore._collection.count() if hasattr(vectorstore, '_collection') else 0,
        "active_sessions": len(HISTORY),
        "total_unanswered": unanswered_count,
        "cache_stats": expand_query.cache_info() if hasattr(expand_query, 'cache_info') else None,
        "token_usage": token_stats,
        "models": {
            "fast": MODEL_FAST,
            "smart": MODEL_SMART,
            "fallback": MODEL_FALLBACK
        }
    })

@app.route("/reset-tokens", methods=["POST"])
def reset_tokens():
    """
    Reset token tracker (untuk testing).
    PERINGATAN: Ini hanya reset tracking lokal, bukan reset quota Groq.
    """
    if os.getenv("DEBUG", "false").lower() != "true":
        return jsonify({"error": "Only available in debug mode"}), 403
    
    global token_tracker
    token_tracker = TokenTracker()
    return jsonify({"status": "success", "message": "Token tracker reset"})

# ── Graceful shutdown ─────────────────────────────────────────
import atexit

def cleanup():
    """Cleanup saat server shutdown"""
    print("\n[SHUTDOWN] Menyimpan state...")
    token_tracker._save_usage()
    print("[SHUTDOWN] Selesai.")

atexit.register(cleanup)

# ── Main entry point ──────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"\n🌐 Server berjalan di http://0.0.0.0:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📚 Vector store: {PERSIST_DIR}")
    print(f"🤖 Model cepat (query expansion): {MODEL_FAST}")
    print(f"🧠 Model pintar (generate jawaban): {MODEL_SMART}")
    print(f"🔄 Model fallback: {MODEL_FALLBACK}")
    print(f"💬 Max history per session: {MAX_HISTORY_LENGTH} pesan")
    print(f"📊 Token limits: {TOKEN_LIMITS}\n")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        threaded=True
    )