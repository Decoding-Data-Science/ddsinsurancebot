# app.py — Insurance Q&A (RAG) with system prompt + simple config
import os
import gradio as gr
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# --- System Prompt (polite + answer-from-document constraint) ---
SYSTEM_PROMPT = """You are Aisha, a polite and professional Insurance assistant.
Answer ONLY using the information found in the indexed insurance document(s).
If the answer is not in the document(s), say: "I couldn’t find that in the document."
Keep responses concise, helpful, and courteous.
"""

# ===== Minimal CONFIG (only necessary keys) =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY or OPENAI_API_KEY (set them in Space → Settings → Variables).")

DATA_DIR = "data"                         # Put insurance docs here (e.g., data/insurance.pdf)
LOGO_PATH = os.path.join(DATA_DIR, "dds_logo.png")  # Mandatory logo
if not os.path.exists(LOGO_PATH):
    raise RuntimeError("Logo not found: data/dds_logo.png.png (commit it to your Space repo).")

EMBED_MODEL = "text-embedding-3-small"    # 1536-dim
LLM_MODEL   = "gpt-4o-mini"
TOP_K       = 4                            # internal similarity_top_k

# ===== LlamaIndex / Pinecone (simple, fixed serverless: aws/us-east-1) =====
Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
Settings.llm = OpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, system_prompt=SYSTEM_PROMPT)

pc = Pinecone(api_key=PINECONE_API_KEY)
def ensure_index(name: str, dim: int = 1536):
    names = [i["name"] for i in pc.list_indexes()]
    if name not in names:
        pc.create_index(
            name=name, dimension=dim, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(name)

# Fixed index name for simplicity
pinecone_index = ensure_index("dds-insurance-index", dim=1536)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

def bootstrap_index():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError("No 'data/' directory found. Commit your documents to data/ in the Space repo.")
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    if not docs:
        raise RuntimeError("No documents found in data/. Add e.g., data/insurance.pdf")
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(docs, storage_context=storage_ctx, show_progress=True)

bootstrap_index()

def answer(query: str) -> str:
    if not query.strip():
        return "Please enter a question (or select one from the FAQ list)."
    index = VectorStoreIndex.from_vector_store(vector_store)
    resp = index.as_query_engine(similarity_top_k=TOP_K).query(query)
    return str(resp)

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

def use_faq(selected_faq: str, free_text: str):
    prompt = (selected_faq or "").strip() or (free_text or "").strip()
    if not prompt:
        return "", "Please select a FAQ or type your question."
    return prompt, answer(prompt)

# ===== UI =====
CSS = """
.header { display:flex; flex-direction:column; align-items:center; gap:6px; }
.logo img { width:300px; height:300px; object-fit:contain; }  /* fixed 300x300 */
.title { text-align:center; font-weight:700; font-size:1.4rem; margin:6px 0 0 0; }
.subnote { text-align:center; margin-top:-2px; opacity:0.8; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("<div class='header'>")
            gr.Image(value=LOGO_PATH, show_label=False, elem_classes=["logo"])
            gr.Markdown(
                "<h1 class='title'>DDS Insurance Q&A — RAG Assistant</h1>"
                "<p class='subnote'>Answers strictly from your insurance document(s)</p>"
            )
            gr.Markdown("</div>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Ask from Frequently Asked Questions")
            faq = gr.Dropdown(choices=FAQS, value=FAQS[0], label="Select a common question")

            gr.Markdown("### Or type your question")
            user_q = gr.Textbox(
                label="Your question",
                placeholder="e.g., What is covered under outpatient benefits?",
                lines=2
            )
            ask_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            chosen_prompt = gr.Textbox(label="Query sent", interactive=False)
            answer_box = gr.Markdown()

    ask_btn.click(use_faq, inputs=[faq, user_q], outputs=[chosen_prompt, answer_box])

if __name__ == "__main__":
    demo.launch()
