import os
import modal

app = modal.App("rag-server-inspector")
rag_volume = modal.Volume.from_name("rag-data-vol")

image = modal.Image.debian_slim(python_version="3.10").pip_install("qdrant-client")

@app.function(
    image=image,
    volumes={"/data": rag_volume},
)
def inspect_data():
    print("\n--- Listing /data contents ---")
    if os.path.exists("/data"):
        print(os.listdir("/data"))
    
    print("\n--- Listing /data/data/vector_db contents ---")
    vdb_path = "/data/data/vector_db"
    if os.path.exists(vdb_path):
        print(os.listdir(vdb_path))
    else:
        print(f"Directory {vdb_path} not found.")

@app.local_entrypoint()
def main():
    inspect_data.remote()