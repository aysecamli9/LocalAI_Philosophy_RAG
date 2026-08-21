
# Philosophy RAG

A local Retrieval-Augmented Generation (RAG) system for answering
philosophical questions using a curated collection of philosophical texts.

The system retrieves relevant passages from the provided documents
and uses a locally running language model to generate answers grounded
in those passages.

---

## Features

- Local Retrieval-Augmented Generation (RAG) pipeline
- Philosophical document retrieval
- Semantic search using embeddings
- SQLite-based vector storage
- Multi-philosopher question support
- Comparative question support
- Source document and page tracking
- Hallucination / out-of-scope question handling
- Local LLM inference
- Performance evaluation
- 15-question evaluation set
- Streamlit-based user interface

---

## Architecture

The system follows the pipeline below:

```text
Philosophical Documents
        ↓
Text Extraction
        ↓
Chunking
        ↓
Embedding Generation
        ↓
SQLite Database
        ↓
Query Embedding
        ↓
Semantic Retrieval
        ↓
Relevant Context
        ↓
Local LLM
        ↓
Answer + Sources
````

---

## Project Structure

```text
LocalAI/
│
├── documents/
│   └── Philosophical source PDFs
│
├── extracted/
│   └── Extracted document text
│
├── chunks/
│   └── Processed document chunks
│
├── Evaluation/
│   └── LocalAI Test Cevapları.pdf
│
├── app.py
├── main.py
├── retrieve.py
├── embed.py
├── chunk.py
├── database.py
├── ingest.py
│
├── philosophy_rag.db
├── embedded_chunks.jsonl
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Document Ingestion

The philosophical source documents are processed and converted into
structured text data.

### 2. Chunking

The extracted texts are divided into smaller chunks so that relevant
passages can be retrieved efficiently.

### 3. Embeddings

Each document chunk is converted into a vector representation using
the local embedding model.

### 4. Retrieval

When a user asks a question, the question is converted into an embedding
and compared against the stored document chunks using cosine similarity.

The most relevant chunks are selected as context for the language model.

### 5. Generation

The retrieved context is passed to a locally running language model.

The model generates an answer based on the retrieved documents rather
than relying solely on its pretrained knowledge.

### 6. Source Attribution

The system reports the source document and page number of the retrieved
chunks used during the response.

---

## Supported Questions

The evaluation set contains different types of questions:

* Single-philosopher questions
* Comparative questions
* Questions involving multiple philosophers
* Questions explicitly supported by the document collection
* Out-of-scope / hallucination-check questions

### Example

**Question:**

> What is the state of nature like according to Hobbes?

The system retrieves relevant passages from the Hobbes documents and
generates an answer grounded in those passages.

---

## Hallucination Handling

The system is designed to avoid generating unsupported information
when the required information is not present in the document collection.

For example:

> According to Socrates, what was his favorite programming language?

If the documents do not contain information relevant to this question,
the system should indicate that there is not enough information in the
knowledge base instead of inventing an answer.

This behavior was also tested using out-of-scope questions in the
evaluation set.

---

## Evaluation

The system was evaluated using a set of **15 test questions** covering:

* Single-philosopher questions
* Comparative questions
* Multi-philosopher questions
* Document-supported questions
* Hallucination-check / out-of-scope questions

The evaluation examined:

* Retrieval relevance
* Answer quality
* Multi-philosopher retrieval
* Out-of-scope question handling
* Source and page attribution
* Response time

### Full Evaluation Results

The complete set of test questions and generated answers is available here:

[View Full Evaluation Results](Evaluation/LocalAI%20Test%20Cevaplar%C4%B1.pdf)

---

## Performance

Performance was measured across the 14-question evaluation set.

| Metric              |  Average |
| ------------------- | -------: |
| Retrieval Time      |  0.638 s |
| Generation Time     | 14.486 s |
| Total Response Time | 15.124 s |

Retrieval remained relatively stable across the evaluation set, while
generation time varied depending on the complexity of the question and
the amount of context required.

---

## Example Output

### Question

> What is the state of nature like according to Hobbes?

### Answer

> The state of nature, according to Hobbes, is a condition of perpetual
> war of every man against every man, where there is no common power to
> keep individuals in awe. In this state, there is no justice or injustice
> because there is no common authority or law.

### Sources

* Hobbes.pdf — Page 14
* Hobbes.pdf — Page 13
* Hobbes.pdf — Page 12

---

## User Interface

The project includes a Streamlit-based web interface for interacting
with the Philosophy RAG system.

Users can enter a philosophical question and receive:

* A generated answer
* Retrieved source information
* Source document names
* Page numbers

The interface provides a simple way to interact with the local RAG
pipeline without using the command line.

---

## Limitations

The system's answer quality depends on the quality and coverage of the
retrieved document chunks.

Comparative questions involving multiple philosophers can be more
challenging because relevant passages must be retrieved for each
philosopher.

The system also depends on the local embedding model and local language
model used for retrieval and generation.

Retrieval quality may vary depending on how well the question matches
the available document chunks.

---

## Technologies

* Python
* Retrieval-Augmented Generation (RAG)
* Local LLM inference
* Embedding models
* Semantic similarity search
* SQLite
* NumPy
* OpenAI-compatible local API
* Streamlit

---

## Models

The project uses locally running models through Microsoft Foundry Local.

### Language Model

```text
phi-4-mini
```

### Embedding Model

```text
qwen3-embedding-0.6b
```

Both models are used locally rather than relying on a cloud-based
LLM API for inference.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aysecamli9/LocalAI_Philosophy_RAG.git
cd LocalAI_Philosophy_RAG
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Microsoft Foundry Local must also be installed and running with the
required local models.

---

## Running the Project

Before running the application, make sure **Microsoft Foundry Local** is installed and the required models are loaded.

### 1. Load the Language Model

Load `phi-4-mini`:

```powershell
foundry model load phi-4-mini
```

### 2. Load the Embedding Model

Load `qwen3-embedding-0.6b`:

```powershell
foundry model load qwen3-embedding-0.6b
```

Both models should be loaded before running the application.

### 3. Configure the Foundry Local URL

Make sure the `FOUNDRY_URL` in **both `main.py` and `retrieve.py`** points to your local Foundry Local endpoint:

```python
FOUNDRY_URL = "http://127.0.0.1:50021/v1"
```

If Foundry Local uses a different port on your system, update the URL accordingly in both files.

### 4. Run the Command-Line Application

```bash
python main.py
```

The application will prompt the user for a philosophical question.

### 5. Run the Streamlit Interface

```bash
streamlit run app.py
```

The Streamlit application will then open in a web browser.

---

## Document Processing Pipeline

If the document collection needs to be rebuilt, the processing pipeline
consists of the following stages:

```text
Documents
   ↓
ingest.py
   ↓
extracted/
   ↓
chunk.py
   ↓
chunks/
   ↓
embed.py
   ↓
embedded vectors
   ↓
philosophy_rag.db
```

The resulting database is then used by the retrieval component.

---

## Demo

A short demonstration video of the Philosophy RAG system is added
here.

[Watch the demo video](#)

---

## Author

**Ayşe Çamlı**
ayse.camli@sabanciuniv.edu
Sabancı University — Computer Engineering Student

```
