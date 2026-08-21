import sqlite3
import unicodedata
import re
import difflib
import numpy as np
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_FILE = "philosophy_rag.db"

EMBEDDING_MODEL = "qwen3-embedding-0.6b"
LLM_MODEL = "phi-4-mini"

# Foundry Local adresi
FOUNDRY_URL = "http://127.0.0.1:49892/v1"

# Normal sorularda alınacak maksimum sonuç
TOP_K = 5

# Karşılaştırmalı sorularda filozof başına sonuç
PER_PHILOSOPHER_K = 3

# Retrieval için minimum similarity threshold
MIN_SIMILARITY = 0.40

# LLM'e gönderilecek maksimum chunk (tek filozof sorularında)
MAX_CONTEXT_CHUNKS = 5

# Karşılaştırma sorularında filozof başına context'e giren chunk sayısı
MAX_CONTEXT_CHUNKS_PER_PHILOSOPHER = 3


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI(
    base_url=FOUNDRY_URL,
    api_key="local"
)


# ============================================================
# STRING NORMALIZATION & PHILOSOPHER LIST
# ============================================================

def normalize_text(text: str) -> str:
    """Türkçe ve diğer aksanlı karakterleri arama için normalize eder."""
    text = text.replace("I", "i").replace("İ", "i").replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


# NOT: Kavramsal alias'lar (örn. "state of nature", "the prince",
# "inequality") isim geçmeyen ama konu bazlı sorularda doğru filozofu
# tespit etmeyi sağlıyor. Dikkat: bazı alias'lar geniş kapsamlı olabilir
# ("prince", "inequality" gibi tek başına genel kelimeler) - eğer ileride
# yanlış filozof tespiti (false positive) görülürse, bu alias'ları daha
# spesifik ifadelerle (örn. "prince" yerine "the prince by machiavelli")
# değiştirmek gerekebilir.
PHILOSOPHERS = {
    "socrates": ["socrates", "sokrates", "apology", "gadfly"],
    "plato": ["plato", "eflatun", "republic", "meno"],
    "aristotle": ["aristotle", "aristoteles", "nicomachean"],
    "epicurus": ["epicurus", "epikur", "epikuros", "hedone", "ataraxia", "aponia"],
    "epictetus": ["epictetus", "epiktetos", "stoic", "stoicism"],
    "hobbes": ["hobbes", "leviathan", "state of nature"],
    "rousseau": ["rousseau", "inequality", "social contract", "amour de soi", "amour-propre"],
    "descartes": ["descartes", "meditations", "cogito", "cartesian"],
    "hume": ["hume", "enquiry", "treatise"],
    "machiavelli": ["machiavelli", "makyavel", "the prince", "prince"],
    "augustine": ["augustine", "augustinus", "confessions"],
    "montaigne": ["montaigne", "essays"],
    "fricker": ["fricker", "testimonial injustice", "hermeneutical injustice"]
}


# ============================================================
# PHILOSOPHER DETECTION
# ============================================================

def detect_philosophers(query: str):
    query_norm = normalize_text(query)
    detected = []

    for philosopher, aliases in PHILOSOPHERS.items():
        for alias in aliases:
            if normalize_text(alias) in query_norm:
                detected.append(philosopher)
                break

    return detected


# ============================================================
# GET EMBEDDING
# ============================================================

def get_embedding(text: str, is_query: bool = False) -> np.ndarray:
    input_text = text
    if is_query:
        input_text = (
            "Instruct: Given a philosophy search query, "
            "retrieve relevant passages that answer the query.\n"
            f"Query: {text}"
        )

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=input_text
    )

    return np.array(
        response.data[0].embedding,
        dtype=np.float32
    )


# ============================================================
# LOAD DATABASE
# ============================================================

def load_chunks():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            id,
            source,
            file_type,
            page,
            chunk_id,
            text,
            embedding
        FROM chunks
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


# ============================================================
# ROW -> RESULT
# ============================================================

def row_to_result(row, similarity: float) -> dict:
    return {
        "id": row[0],
        "source": row[1],
        "file_type": row[2],
        "page": row[3],
        "chunk_id": row[4],
        "text": row[5],
        "similarity": float(similarity)
    }


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(query_embedding: np.ndarray, rows: list) -> list:
    if not rows:
        return []

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []

    query_embedding = query_embedding / query_norm

    embeddings = np.array(
        [np.frombuffer(row[6], dtype=np.float32) for row in rows],
        dtype=np.float32
    )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized_embeddings = embeddings / norms

    similarities = np.dot(normalized_embeddings, query_embedding)

    results = []
    for index, similarity in enumerate(similarities):
        if similarity < MIN_SIMILARITY:
            continue
        results.append(row_to_result(rows[index], similarity))

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


# ============================================================
# PHILOSOPHER SOURCE MATCHING
# ============================================================

def source_belongs_to_philosopher(source: str, philosopher: str) -> bool:
    source_norm = normalize_text(source)
    aliases = PHILOSOPHERS.get(philosopher, [])
    return any(normalize_text(alias) in source_norm for alias in aliases)


# ============================================================
# REMOVE DUPLICATE SOURCE/PAGE
# ============================================================

def remove_duplicate_pages(results: list) -> list:
    unique = []
    seen = set()

    for result in results:
        key = (result["source"], result["page"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)

    return unique


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query: str):
    query_embedding = get_embedding(query, is_query=True)
    rows = load_chunks()

    if not rows:
        return [], []

    detected_philosophers = detect_philosophers(query)

    # Multi-Philosopher Soruları
    if len(detected_philosophers) >= 2:
        print(f"Detected philosophers: {', '.join(detected_philosophers)}")
        all_results = []

        for philosopher in detected_philosophers:
            philosopher_rows = [
                row for row in rows
                if source_belongs_to_philosopher(row[1], philosopher)
            ]

            philosopher_results = vector_search(query_embedding, philosopher_rows)
            philosopher_results = remove_duplicate_pages(philosopher_results)
            philosopher_results = philosopher_results[:PER_PHILOSOPHER_K]

            for result in philosopher_results:
                result["detected_philosopher"] = philosopher
                all_results.append(result)

        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results, detected_philosophers

    # Tek Filozof Soruları
    if len(detected_philosophers) == 1:
        philosopher = detected_philosophers[0]
        print(f"Detected philosopher: {philosopher}")

        philosopher_rows = [
            row for row in rows
            if source_belongs_to_philosopher(row[1], philosopher)
        ]

        results = vector_search(query_embedding, philosopher_rows)
        results = remove_duplicate_pages(results)

        for result in results:
            result["detected_philosopher"] = philosopher

        return results[:TOP_K], detected_philosophers

    # Genel / Filozof İsmi Geçmeyen Sorular
    print("No specific philosopher detected.")
    results = vector_search(query_embedding, rows)
    results = remove_duplicate_pages(results)

    return results[:TOP_K], detected_philosophers


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text: str) -> str:
    text_lower = text.lower()
    turkish_characters = "çğıöşü"

    if any(char in text_lower for char in turkish_characters):
        return "Turkish"

    turkish_markers = {
        "ve", "ile", "nedir", "nasil", "gore", "hakkinda", "insan",
        "insanin", "olan", "olanlar", "arasindaki", "fark", "farklar",
        "savunmasinda", "devlet", "ahlak", "mutluluk", "bilgi", "nicin", "neden"
    }

    norm_words = (
        normalize_text(text)
        .replace("?", "")
        .replace(",", "")
        .replace(".", "")
        .split()
    )

    if any(word in turkish_markers for word in norm_words):
        return "Turkish"

    return "English"


# ============================================================
# REPETITION GUARD
# ============================================================

def cut_repetition(text: str) -> str:
    """
    Aynı 3-5 kelimelik öbek art arda 3+ kere tekrar ediyorsa,
    metni tekrarın başladığı noktadan keser. Küçük modellerde
    görülen 'the im the im the im ...' tarzı döngüleri temizler.
    """
    words = text.split()
    for n in (3, 4, 5):
        i = 0
        while i + n * 3 <= len(words):
            chunk = words[i:i + n]
            if words[i + n:i + 2 * n] == chunk and words[i + 2 * n:i + 3 * n] == chunk:
                return " ".join(words[:i]).strip()
            i += 1
    return text


def cut_duplicate_sentences(text: str) -> str:
    """
    Cümle/paragraf seviyesinde tekrarları yakalar. Küçük modellerde
    görülen 'aynı paragrafı 2 kez üretme' durumunu, n-gram tekrar
    dedektörü (3x art arda gerektirir) yakalayamaz çünkü genelde
    sadece 2 kez tekrarlanır. Burada her cümleyi daha önce görülen
    cümlelerle karşılaştırıp, yüksek benzerlik (>=0.85) bulunduğu an
    metni o noktadan keser.
    """
    sentences = re.split(r'(?<!\d)(?<=[.!?])\s+', text.strip())

    seen = []
    kept = []

    for sentence in sentences:
        normalized = " ".join(sentence.lower().split())

        # Çok kısa cümleler (bağlaçlar vs.) yanlış pozitif üretebilir, atla
        if len(normalized.split()) < 6:
            kept.append(sentence)
            seen.append(normalized)
            continue

        is_duplicate = False
        for prev in seen:
            ratio = difflib.SequenceMatcher(None, normalized, prev).ratio()
            if ratio >= 0.85:
                is_duplicate = True
                break

        if is_duplicate:
            break

        kept.append(sentence)
        seen.append(normalized)

    return " ".join(kept).strip()


def strip_document_references(text: str) -> str:
    """
    Modelin cevabına sızdırdığı kaynak referanslarını temizler.
    İki formu da kapsar:
    - Parantez/köşeli parantez içinde: "(Document 1)", "[Source 2]"
    - Cümle içinde çıplak geçen: "Document 1 and 2 clearly state...",
      "especially SOURCE 3, where..."
    """
    # 1) Parantez/köşeli parantez içindeki referanslar
    pattern_bracketed = r'[\(\[]\s*(document|source|kaynak|sayfa|page)\s*\d*\s*[\)\]]'
    cleaned = re.sub(pattern_bracketed, "", text, flags=re.IGNORECASE)

    # 2) Cümle içinde çıplak geçen "Document 1", "SOURCE 3 and 5" gibi ifadeler
    pattern_bare = r'\b(document|source|belge|kaynak)s?\s*\d+(\s*(and|,|ve)\s*\d+)*\b'
    cleaned = re.sub(pattern_bare, "the text", cleaned, flags=re.IGNORECASE)

    # Boşluk/noktalama temizliği
    cleaned = re.sub(r'\s+([,.;:])', r'\1', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)

    # Cümle başına denk gelen "the text" -> "The text" büyük harf düzeltmesi
    cleaned = re.sub(
        r'(^|[.!?]\s+)the text\b',
        lambda m: m.group(1) + "The text",
        cleaned
    )

    return cleaned.strip()


def cut_context_leakage(text: str) -> str:
    """
    Model context bloklarını ('--- SOURCE', 'Document:', vb.)
    cevabın içine sızdırdıysa, sızıntının başladığı noktadan keser.
    """
    markers = ["--- SOURCE", "\nDocument:", "\nKAYNAKLAR", "SOURCE 1", "SOURCE 2"]
    cut_index = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            cut_index = min(cut_index, idx)
    return text[:cut_index].strip()


def cap_sentences(text: str, max_sentences: int) -> str:
    """
    Cevabı en fazla max_sentences cümleyle sınırlar. Model context'i
    doldurmaya devam edip aynı fikri farklı cümlelerle tekrar tekrar
    parafrazladığında (difflib eşiğini geçmeyecek kadar farklı ifadeler
    olduğu için cut_duplicate_sentences bunu yakalayamaz), bu fonksiyon
    cevabı sabit bir uzunlukta tutarak rambling'i sınırlar.
    """
    sentences = re.split(r'(?<!\d)(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences]).strip()


# ============================================================
# GENERATION (TWO-STEP TRANSLATION PIPELINE)
# ============================================================

def generate_answer(query: str, retrieved_chunks: list, detected_philosophers: list) -> str:

    not_found_msg = (
        "The provided documents do not contain enough information "
        "to answer this question."
    )

    if not retrieved_chunks:
        return not_found_msg

    # --------------------------------------------------
    # BASIC RETRIEVAL CHECK
    # --------------------------------------------------

    # Only reject when retrieval itself is genuinely weak.
    if retrieved_chunks[0]["similarity"] < 0.40:
        return not_found_msg

    # --------------------------------------------------
    # DETERMINE QUESTION TYPE
    # --------------------------------------------------

    philosopher_names = ", ".join(detected_philosophers)
    is_comparison = len(detected_philosophers) >= 2

    # --------------------------------------------------
    # CONTEXT SELECTION
    # --------------------------------------------------
    # Karşılaştırma sorularında similarity'e göre kesip bir filozofu
    # context dışında bırakmamak için her filozoftan garanti chunk alıyoruz.

    if is_comparison:
        grouped = {}
        for chunk in retrieved_chunks:
            key = chunk.get("detected_philosopher", "unknown")
            grouped.setdefault(key, []).append(chunk)

        selected_chunks = []
        for philosopher in detected_philosophers:
            phil_chunks = grouped.get(philosopher, [])
            selected_chunks.extend(phil_chunks[:MAX_CONTEXT_CHUNKS_PER_PHILOSOPHER])
    else:
        selected_chunks = retrieved_chunks[:MAX_CONTEXT_CHUNKS]

    context_parts = []

    for i, chunk in enumerate(selected_chunks, start=1):

        clean_text = " ".join(
            chunk["text"].split()
        )[:1200]

        context_parts.append(
            f"""--- SOURCE {i} ---
Document: {chunk["source"]}
Page: {chunk["page"]}

{clean_text}"""
        )

    context = "\n\n".join(context_parts)

    if is_comparison:
        question_instruction = """
    This is a comparison question. Write a short flowing paragraph of
    EXACTLY TWO sentences, no bullets, no numbering, no list format:

    - First sentence: explain the FIRST philosopher's view, answering
      specifically what the question asks. Start the sentence directly
      with the philosopher's name used naturally as the subject (e.g.
      "Hobbes argues that..."), not as a label or heading before a colon.
    - Second sentence: explain the SECOND philosopher's view the same
      way, opening with a short contrast phrase such as "In contrast,"
      or "Rousseau, on the other hand,", again answering specifically
      what the question asks.

    Do NOT add a third sentence, a concluding summary, or any statement
    about "the main difference" — the two sentences placed side by side
    are the complete answer. Do not use dashes, bullets, or numbers
    anywhere in the answer.
    """
    else:
        question_instruction = """
This is a single-philosopher question.

Focus on the philosopher relevant to the question and use the
most relevant information from the context.
"""

    # --------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------

    system_prompt = f"""
You are an academic philosophy question-answering assistant.

Answer the QUESTION using ONLY the provided DOCUMENT CONTEXT.

Detected philosopher(s):
{philosopher_names}

{question_instruction}

RULES:

1. Use only information supported by the provided context.
2. Do not use outside knowledge.
3. Answer the question directly.
4. Combine information from multiple context chunks when necessary.
5. If the context contains the answer indirectly or across multiple chunks,
   infer the answer only from those supported details.
6. Do not require the exact wording of the question to appear in the documents.
7. If the question is a comparison, write exactly two sentences in a
   single flowing paragraph — one per philosopher, no bullets, no
   numbering, no heading with the philosopher's name before a colon,
   and no third/concluding sentence.
8. If the question contains a false premise, or asks about something
   genuinely absent from the context (anachronistic topics, claims the
   philosopher never made, etc.), reject it in ONE short sentence only.
   Do not explain at length why it is absent, do not speculate about
   loosely related content, and do not list what the context does
   discuss instead.
9. Only say that there is not enough information when the core information
   needed to answer the question is genuinely absent from the context.
   If the context contains even partial or indirect evidence relevant to
   the question, use it to give the best possible answer instead of
   declining. When you do decline, keep it to ONE short sentence.
10. Do not repeat sentences, phrases, or ideas.
11. Do not copy corrupted or malformed text from the documents.
12. Rephrase the information naturally.
13. Answer in clear, natural English.
14. For single-philosopher questions, keep the answer to 2-3 sentences.
    For comparison questions, write EXACTLY two sentences total (one
    per philosopher) in plain paragraph form — no bullets, no
    numbering, no extra sentences before or after.
15. Do not mention sources, pages, retrieval, context, or document metadata.
16. Never output text that looks like "--- SOURCE", "Document:", or "Page:".
17. Do not mention the same idea more than once.
18. Never reproduce phrases such as "Montaigne also found",
    "The documents state", or similar repetitive constructions.
19. Never output document names, chapter names, source labels,
    or metadata even if they appear in the context.
20. Before finishing, check the answer for repetition and remove
    any repeated idea.
21. Stop writing as soon as the answer is complete. Do not continue
    with unrelated or repeated content.
22. Never write parenthetical citations such as "(Document 1)",
    "(Source 2)", or "(Page 3)". Never write bare references either,
    such as "Document 1 and 2 state..." or "as seen in SOURCE 3".
    Integrate evidence into the prose without naming or numbering
    where it came from.
23. Do not list multiple quotes, examples, or Latin phrases one
    after another. Pick at most ONE illustrative example and
    synthesize the rest into a single coherent explanation.
24. Never generate the same sentence, or a near-identical
    rephrasing of a sentence, more than once in the answer.
25. State each distinct idea or claim only ONCE. Do not restate the
    same point again later in the answer using different wording,
    even if it is phrased differently. Plan the answer's key points
    first, then write each one a single time.
"""

    # --------------------------------------------------
    # USER PROMPT
    # --------------------------------------------------

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

QUESTION:

{query}

ANSWER:
"""

    # --------------------------------------------------
    # GENERATION
    # --------------------------------------------------

    # Karşılaştırma soruları iki filozofu da ele almak zorunda olduğu
    # için tek filozof sorularına göre daha fazla token'a ihtiyaç duyar.
    # Not: karşılaştırma formatı artık 1+1+1 cümleye sıkıştırıldığı için
    # eskisi kadar (400) token'a ihtiyaç yok; yine de biraz pay bırakıldı.
    max_tokens = 340 if is_comparison else 260

    response = client.chat.completions.create(
        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1,
        top_p=0.9,

        frequency_penalty=0.6,
        presence_penalty=0.2,

        max_tokens=max_tokens,

        # Model context bloklarını taklit etmeye başlarsa üretimi durdur.
        stop=["--- SOURCE", "\nDocument:", "\nKAYNAKLAR", "\nSOURCE "]
    )

    answer = response.choices[0].message.content

    if not answer:
        return not_found_msg

    answer = answer.strip()

    # --------------------------------------------------
    # OUTPUT CLEANING
    # --------------------------------------------------

    unwanted_prefixes = [
        "ANSWER:",
        "Answer:",
        "CEVAP:",
        "Cevap:"
    ]

    for prefix in unwanted_prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    # Context sızıntısı ve tekrar döngüsü temizliği (stop sequence'e
    # rağmen yine de sızıntı olursa ek güvenlik ağı).
    answer = cut_context_leakage(answer)
    answer = cut_repetition(answer)          # kelime/n-gram döngüleri (3x tekrar)
    answer = cut_duplicate_sentences(answer)  # cümle/paragraf düzeyinde tekrar (2x tekrar dahil)
    answer = strip_document_references(answer)  # "(Document 1)" / "SOURCE 3" gibi sızıntılar

    # Fikir tekrarı / rambling sınırlaması. Karşılaştırma formatı artık
    # 1+1+1 cümle istediği için sınırı da buna göre dar tutuyoruz
    # (küçük bir tolerans payıyla), tek filozof sorularında 4 cümle.
    if is_comparison:
        max_sentences = len(detected_philosophers)
    else:
        max_sentences = 4
    answer = cap_sentences(answer, max_sentences)

    answer = answer.strip()

    return answer if answer else not_found_msg

def print_sources(retrieved_chunks: list):
    max_shown = (
        MAX_CONTEXT_CHUNKS_PER_PHILOSOPHER * 3
        if retrieved_chunks and "detected_philosopher" in retrieved_chunks[0]
        else MAX_CONTEXT_CHUNKS
    )
    selected = retrieved_chunks[:max_shown]
    print("\nKAYNAKLAR\n")
    seen = set()
    for chunk in selected:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            print(f"- {chunk['source']} (Page {chunk['page']})")
            seen.add(key)


# ============================================================
# MAIN
# ============================================================

def main():

    query = input("\nSorunuzu yazın: ").strip()

    if not query:
        print("\nSoru boş bırakılamaz.")
        return

    print("\nAranıyor...")

    try:
        retrieved_chunks, detected_philosophers = retrieve(query)
    except Exception as error:
        print(f"\n[HATA] Retrieval esnasında hata oluştu: {error}")
        return

    print("ANSWER")

    try:
        answer = generate_answer(
            query,
            retrieved_chunks,
            detected_philosophers
        )
    except Exception as error:
        print(f"\n[ERROR] Generation failed: {error}")
        return

    print(answer)

    print_sources(retrieved_chunks)


if __name__ == "__main__":
    main()