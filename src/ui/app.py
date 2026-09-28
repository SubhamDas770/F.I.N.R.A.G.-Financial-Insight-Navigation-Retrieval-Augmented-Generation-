import os
import sys
import uuid
import gradio as gr

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Ensure src module can be imported cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.vector_store import RAGVectorStore
from src.retrieval.cache import RedisSemanticCache
from src.memory.chat_history import RedisChatMemory
from src.retrieval.orchestrator import RAGOrchestrator


# -------------------------------------------------------------
# 1. Initialize Pipeline Backend
# -------------------------------------------------------------
def initialize_pipeline():
    print("🚀 Initializing Enterprise RAG Services...")
    
    # Vector store with Qdrant and local persistence
    vector_store = RAGVectorStore(
        collection_name="enterprise_docs",
        qdrant_path="./data/vector_db"
    )
    
    # Redis Semantic Cache
    cache = RedisSemanticCache(
        similarity_threshold=0.90,
        ttl_seconds=86400
    )
    
    # Redis Session Chat Memory
    memory = RedisChatMemory(
        max_tokens=2000,
        ttl_seconds=86400
    )
    
    # Orchestrator binding all services
    orchestrator = RAGOrchestrator(
        vector_store=vector_store,
        cache=cache,
        memory=memory
    )
    
    return orchestrator

# Initialize global backend instance
rag_backend = initialize_pipeline()


# -------------------------------------------------------------
# 2. UI Event Handlers
# -------------------------------------------------------------
def handle_user_query(user_message: str, history: list, session_id: str):
    """
    Executes the user query through RAGOrchestrator and updates UI panels.
    """
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        return "", history, {}, "0 tokens", 0.0, "N/A", "0.0s"

    # Generate or reuse session ID for chat memory scoping
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Execute RAG Pipeline
        result = rag_backend.ask(query=user_message, session_id=session_id)

        # Extract results
        bot_answer = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]
        tokens_used = result["prompt_tokens"]
        confidence = result["confidence_score"]
        cache_status = result["cache_status"]
        latency = f"{result['latency_seconds']}s"
    except Exception as e:
        import traceback
        traceback.print_exc()
        bot_answer = f"⚠️ An error occurred during retrieval/generation:\n\n`{str(e)}`"
        retrieved_chunks = []
        tokens_used = "0 tokens"
        confidence = 0.0
        cache_status = "ERROR"
        latency = "0.0s"

    # Gradio 6.28 uses messages format by default: list of dicts
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_answer})

    # Return cleared input box + updated chatbot + HUD outputs
    return (
        "",               # Clear query text box
        history,          # Updated chat history
        retrieved_chunks, # JSON chunk view
        tokens_used,      # Token count string
        confidence,       # Reranker score
        cache_status,     # Exact/Semantic/Miss
        latency           # Pipeline execution latency
    )


def handle_clear_chat(session_id: str):
    """Clears frontend chat window and resets Redis session memory."""
    if session_id:
        rag_backend.memory.clear_session(session_id)
    new_session = str(uuid.uuid4())
    return [], {}, "0 tokens", 0.0, "N/A", "0.0s", new_session


# -------------------------------------------------------------
# 3. Custom Aesthetic Styles
# -------------------------------------------------------------
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #f8fafc;
}
.header-container {
    text-align: center;
    margin-bottom: 1.5rem !important;
}
.chat-box {
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0 !important;
    background: #ffffff;
}
.hud-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0;
}
.metric-badge {
    font-weight: 600;
}
"""


# -------------------------------------------------------------
# 4. Build Gradio Layout
# -------------------------------------------------------------
with gr.Blocks(title="Enterprise RAG Portal") as demo:

    # Hidden State for maintaining unique session UUIDs
    session_state = gr.State(value=lambda: str(uuid.uuid4()))

    gr.Markdown(
        """
        # 🏢 Enterprise RAG Knowledge Portal
        ### *High-Precision Context Retrieval & QLoRA Llama 3 Fine-Tuned Intelligence*
        """,
        elem_classes=["header-container"]
    )

    with gr.Row():
        # ==========================================
        # LEFT PANEL: Chat Interface (Scale 2 = 66% width)
        # ==========================================
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with Enterprise Knowledge Base")
            
            chatbot = gr.Chatbot(
                label="Llama 3 RAG Assistant",
                height=580,
                elem_classes=["chat-box"]
            )
            

            with gr.Row():
                query_input = gr.Textbox(
                    show_label=False,
                    placeholder="Ask a detailed question regarding your enterprise documents...",
                    container=False,
                    scale=4
                )
                submit_btn = gr.Button("Send Question 🚀", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("Clear Chat History", variant="stop", size="sm")

        # ==========================================
        # RIGHT PANEL: Retrieval Inspector HUD (Scale 1 = 33% width)
        # ==========================================
        with gr.Column(scale=1, elem_classes=["hud-card"]):
            gr.Markdown("### 🔍 Real-Time Retrieval Inspector")
            gr.Markdown("Inspect exact document chunks, token usage, and cache triggers sent to Llama 3.")

            with gr.Row():
                cache_badge = gr.Textbox(label="Cache Status", value="N/A", interactive=False)
                latency_badge = gr.Textbox(label="Latency", value="0.0s", interactive=False)

            with gr.Row():
                token_badge = gr.Textbox(label="Token Usage", value="0 tokens", interactive=False)
                confidence_score = gr.Number(label="Reranker Score", value=0.0, precision=4, interactive=False)

            gr.Markdown("#### 📄 Fetched Parent Document Chunks")
            chunks_json = gr.JSON(
                label="Top-K Reranked Context",
                value=[],
                show_label=False
            )

    # ==========================================
    # 5. Event Binding
    # ==========================================
    # User submits question via ENTER key
    query_input.submit(
        fn=handle_user_query,
        inputs=[query_input, chatbot, session_state],
        outputs=[
            query_input,
            chatbot,
            chunks_json,
            token_badge,
            confidence_score,
            cache_badge,
            latency_badge
        ]
    )

    # User submits question via Send button
    submit_btn.click(
        fn=handle_user_query,
        inputs=[query_input, chatbot, session_state],
        outputs=[
            query_input,
            chatbot,
            chunks_json,
            token_badge,
            confidence_score,
            cache_badge,
            latency_badge
        ]
    )

    # Clear history button reset
    clear_btn.click(
        fn=handle_clear_chat,
        inputs=[session_state],
        outputs=[
            chatbot,
            chunks_json,
            token_badge,
            confidence_score,
            cache_badge,
            latency_badge,
            session_state
        ]
    )


if __name__ == "__main__":
    import socket
    def find_free_port(start=7860, end=7870):
        for port in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise OSError(f"No free port found in range {start}-{end}")

    free_port = find_free_port()
    print(f"Starting Gradio on port {free_port}...")
    demo.queue().launch(server_name="localhost", server_port=free_port, share=True, show_error=True)