DDS Insurance Q&A — RAG Assistant (Pinecone + OpenAI + Gradio)

Summary: A beginner-friendly, document-grounded insurance bot that you can replicate and deploy on Hugging Face Spaces. It answers only from your uploaded insurance documents using LlamaIndex + Pinecone (serverless) + OpenAI with a simple, polite system prompt.

What You’ll Get

Deployed Space URL you can share.

Grounded answers (no docs → the bot politely says it can’t find it).

Simple UI with an FAQ dropdown + free-text question box.

Clean structure designed for easy replication.

Features

Answers strictly from your data/ documents (RAG).

Pinecone serverless index (AWS us-east-1, cosine, 1536-dim).

OpenAI for embeddings (text-embedding-3-small) and LLM (gpt-4o-mini).

Gradio interface with a centered required logo (data/dds_logo.png).

Beginner-friendly defaults and error messages.

Repository Structure
.
├─ data/                     # Your insurance docs + required logo
│  └─ dds_logo.png           # REQUIRED (shown in header)
├─ app.py                    # Main app: indexing + query + Gradio UI
├─ requirements.txt          # Dependencies
└─ README.md                 # This file

Configuration (in app.py)
EMBED_MODEL = "text-embedding-3-small"   # 1536-dim
LLM_MODEL   = "gpt-4o-mini"
TOP_K       = 4                          # retrieval depth


System Prompt (keeps answers grounded + polite):

SYSTEM_PROMPT = """You are Aisha, a polite and professional Insurance assistant.
Answer ONLY using the information found in the indexed insurance document(s).
If the answer is not in the document(s), say: "I couldn’t find that in the document."
Keep responses concise, helpful, and courteous.
"""


FAQ List (editable):

FAQS = [
    "",
    "What benefits are covered under the policy?",
    "How do I file a claim and what documents are required?",
    "What are the exclusions and limitations?",
    "Is pre-authorization needed for hospitalization?",
    "What is the reimbursement timeline?",
    "How are outpatient vs inpatient services handled?",
    "How can I check my network hospitals/clinics?",
    "What is the co-pay or deductible policy?",
]

Deploy to Hugging Face Spaces (Beginner-Friendly)
1) Create a Space

Go to Hugging Face → Spaces → New Space

SDK: Gradio

Visibility/licensing: your choice

2) Add Project Files

Upload these into your Space:

app.py

requirements.txt

README.md

Create folder data/ and upload:

Your insurance documents (PDF/TXT/MD…)

dds_logo.png (mandatory; exact filename)

Tip: Your Space file tree should match the Repository Structure above.

3) Set Secrets (Environment Variables)

In Space → Settings → Variables and secrets, add:

OPENAI_API_KEY → your OpenAI key

PINECONE_API_KEY → your Pinecone key

No legacy Pinecone environment URL needed. This app uses pinecone-client ≥ 5 with serverless.

4) Build & Run

Spaces auto-install from requirements.txt.

Default CPU hardware is fine.

Entry point auto-detected from app.py.

On first start, the app will:

Ensure a Pinecone serverless index:
dds-insurance-index · cosine · 1536-dim · aws/us-east-1

Read and index documents from data/

Launch the Gradio UI

Your deployed link is simply the Space URL once its status is Running.

5) Updating Documents Later

Upload/change files in data/

Click Restart on the Space so it re-indexes your documents

Troubleshooting (Common Issues)

“Missing PINECONE_API_KEY or OPENAI_API_KEY”
Add both secrets in Space → Settings → Variables and secrets.

Pinecone 401 / “Malformed domain”

Ensure you’re on pinecone-client>=5.0.1 (already in requirements.txt).

Use a valid Pinecone API key; no environment URL needed for serverless.

“Logo not found: data/dds_logo.png”
Upload an image named exactly dds_logo.png into the data/ folder.

“No documents found in data/”
Upload at least one doc (PDF/TXT/MD) into data/, then Restart the Space.

OpenAI authorization/rate-limit errors
Confirm key validity and model access; reduce usage if rate-limited.

Slow first load
First run installs dependencies and builds the index; later runs are faster.

Manual Test Checklist

Ask a question clearly answered in your docs → response should quote that knowledge.

Ask something not in your docs → bot should say it can’t find it.

Adjust TOP_K in app.py to see how answer completeness changes.

Requirements (from requirements.txt)
gradio>=4.44.0
pinecone-client>=5.0.1
openai>=1.51.0
llama-index>=0.11.0
llama-index-vector-stores-pinecone>=0.3.0
llama-index-embeddings-openai>=0.3.0
llama-index-llms-openai>=0.2.0
tiktoken>=0.7.0

Customization Ideas

Swap LLMs by editing LLM_MODEL.

Add a file uploader to refresh docs from the UI.

Add metadata filters (e.g., policy type).

Log queries to refine the FAQ list.

License

Add your chosen license (e.g., MIT) as LICENSE.

Acknowledgments

Thanks to LlamaIndex, Pinecone, OpenAI, and Gradio for the tooling that makes this simple and reproducible.
