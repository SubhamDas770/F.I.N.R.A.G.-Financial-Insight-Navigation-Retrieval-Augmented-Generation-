import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Project root: src/training/dataset.py -> parents[2] = project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "processed" / "llama3_training_data.jsonl"

# Note: In a real-world scenario, you would use a stronger LLM (like GPT-4, Claude 3.5, or Llama 3 70B) 
# to generate the questions and answers from the text. Here we provide the generation wrapper.
def generate_synthetic_qa(context: str, chunk_metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Simulates calling an LLM to generate 2-3 Question/Answer pairs based solely on the provided text chunk.
    """
    # TODO: Replace with an actual LLM API call (e.g., OpenAI or local vLLM endpoint)
    # Example Prompt to the LLM: 
    # "Given the following text, generate 3 factual question and answer pairs. 
    # Text: {context}"
    
    # Mocking the LLM output for demonstration
    page = chunk_metadata.get("page_number", "unknown")
    mock_qas = [
        {
            "question": f"What key metric is discussed on page {page}?",
            "answer": f"Based on the provided context, {context[:50]}... is the key metric discussed."
        },
        {
            "question": "Can you summarize the main finding in this section?",
            "answer": f"The main finding is that {context[-50:]}."
        }
    ]
    return mock_qas

class DatasetGenerator:
    def __init__(self, system_prompt: str = None):
        """
        Initializes the generator with a system prompt that defines the Llama 3 agent's persona.
        """
        self.system_prompt = system_prompt or (
            "You are an expert enterprise AI assistant. "
            "Answer the user's questions accurately and concisely based on the internal company knowledge base."
        )

    def generate_qlora_dataset(
        self, 
        parent_chunks: List[Any], 
        output_filepath: str = str(_DEFAULT_OUTPUT)
    ):
        """
        Iterates over parent chunks, generates synthetic QA pairs, and formats them into
        the Llama 3 instruction dataset structure for SFTTrainer.
        """
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset_records = []
        
        print(f"🧠 Generating synthetic QA pairs from {len(parent_chunks)} Parent Chunks...")
        
        for index, parent in enumerate(parent_chunks):
            # 1. Generate Q&A pairs from the chunk text
            qa_pairs = generate_synthetic_qa(parent.text, parent.metadata)
            
            for qa in qa_pairs:
                # 2. Format into Llama 3 expected conversational structure
                record = {
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": qa["question"]},
                        {"role": "assistant", "content": qa["answer"]}
                    ],
                    "metadata": parent.metadata
                }
                dataset_records.append(record)
                
            if (index + 1) % 10 == 0:
                print(f"  Processed {index + 1}/{len(parent_chunks)} chunks...")

        # Shuffle to ensure the model doesn't overfit to document chronological order
        random.shuffle(dataset_records)

        # 3. Write to JSONL
        print(f"💾 Saving {len(dataset_records)} training records to {output_filepath}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for record in dataset_records:
                f.write(json.dumps(record) + "\n")
                
        print("✅ Dataset generation complete.")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project root to sys.path so 'src' is importable when running directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.ingestion.chunker import ParentChunk
    
    # Mock data to test the script
    mock_parents = [
        ParentChunk(
            parent_id="p-123",
            text="The enterprise software division achieved $12.5M ARR in Q3. Serverless infrastructure reduced overhead by 34%.",
            metadata={"source": "q3_report.pdf", "page_number": 4}
        )
    ]
    
    generator = DatasetGenerator()
    # Use _DEFAULT_OUTPUT so the file is always saved inside the project folder
    generator.generate_qlora_dataset(mock_parents, output_filepath=str(_DEFAULT_OUTPUT))