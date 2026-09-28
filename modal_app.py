import modal

app = modal.App("enterprise-rag-app")
rag_volume = modal.Volume.from_name("rag-data-vol")

# Pin stable version of Gradio and FastAPI to prevent blank screen asset bugs
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi==0.115.4",
        "gradio==4.44.1",
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "qdrant-client",
        "sentence-transformers",
        "redis",
        "pypdf",
        "uvicorn"
    )
    .add_local_dir("src", remote_path="/root/src")
)

@app.function(
    image=image,
    volumes={"/data": rag_volume},
    timeout=600,
)
@modal.asgi_app()
def web():
    import sys
    sys.path.append("/root")

    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from src.ui.app import build_ui

    web_app = FastAPI()
    demo = build_ui()
    
    # Enable queuing to prevent websockets/long-polling timeout issues
    demo.queue()

    return mount_gradio_app(app=web_app, blocks=demo, path="/")