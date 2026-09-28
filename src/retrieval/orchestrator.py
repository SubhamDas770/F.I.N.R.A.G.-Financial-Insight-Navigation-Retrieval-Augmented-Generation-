import os
import time
import torch
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load environment variables (.env)
load_dotenv()

from src.retrieval.cache import RedisSemanticCache
from src.retrieval.vector_store import RAGVectorStore
from src.memory.chat_history import RedisChatMemory


class RAGOrchestrator:
    def __init__(
        self,
        vector_store: RAGVectorStore,
        cache: RedisSemanticCache,
        memory: RedisChatMemory,
        reranker_model_name: str = "BAAI/bge-reranker-large",
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        fine_tuned_adapter_path: str = "./data/fine_tuned_llama3/final_adapters",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        print("⚙️ Initializing RAG Pipeline Orchestrator...")
        self.vector_store = vector_store
        self.cache = cache
        self.memory = memory
        self.device = device
        hf_token = os.getenv("HF_TOKEN") or None

        # 1. Load Cross-Encoder Reranker
        print(f"🔄 Loading Reranker Model: {reranker_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name, max_length=512, device=device)

        # 2. Load Tokenizer & Llama 3 LLM
        print(f"🤖 Loading Model: {model_id}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"⚠️ Failed to load tokenizer for {model_id}: {e}")
            self.tokenizer = None

        # Check if CUDA is available for GPU loading; fallback to CPU if local testing without GPU
        if device == "cuda" and self.tokenizer is not None:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    token=hf_token,
                )
                
                # Attach fine-tuned QLoRA adapters if available
                if os.path.exists(fine_tuned_adapter_path):
                    print(f"🎯 Attaching fine-tuned LoRA adapters from: {fine_tuned_adapter_path}")
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, fine_tuned_adapter_path)
            except Exception as e:
                print(f"⚠️ Could not load CUDA model {model_id}: {e}. Falling back to CPU mode.")
                self.model = None
        else:
            print("⚠️ GPU not detected or tokenizer unavailable. Running in CPU context response mode.")
            self.model = None

    def _rerank_chunks(self, query: str, raw_hits: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Passes retrieved vector hits through the Cross-Encoder reranker to assign precise relevance scores.
        """
        if not raw_hits:
            return []

        # Prepare (query, context) pairs for Cross-Encoder scoring (guarantee string types)
        pairs = [(query, str(hit.get("parent_text") or hit.get("child_text") or "")) for hit in raw_hits]
        scores = self.reranker.predict(pairs)

        # Attach reranker scores to hits
        for hit, score in zip(raw_hits, scores):
            hit["rerank_score"] = float(score)

        # Sort descending by cross-encoder relevance score
        reranked_hits = sorted(raw_hits, key=lambda x: x["rerank_score"], reverse=True)
        
        # Deduplicate parent chunks while maintaining top relevance
        seen_parents = set()
        unique_parents = []
        
        for hit in reranked_hits:
            parent_id = hit.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                unique_parents.append(hit)
            if len(unique_parents) >= top_n:
                break

        return unique_parents

    def _generate_response(self, query: str, context_chunks: List[Dict[str, Any]], history: List[Dict[str, str]]) -> Tuple[str, int]:
        """
        Formats prompt using Llama 3 chat template and generates text response.
        """
        # Combine retrieved parent texts into a single context block
        context_str = "\n\n---\n\n".join(
            [f"[Source: {c.get('metadata', {}).get('source', 'Document')} | Page {c.get('metadata', {}).get('page_number', 1)}]\n{c.get('parent_text', '')}" 
             for c in context_chunks]
        )

        system_instruction = (
            "You are an enterprise AI assistant answering questions based strictly on the provided company document context.\n"
            "If the answer cannot be determined from the context, state that clearly.\n\n"
            f"=== DOCUMENT CONTEXT ===\n{context_str}\n========================"
        )

        messages = [{"role": "system", "content": system_instruction}]
        
        # Append prior turn chat history
        messages.extend(history)
        
        # Append current user question
        messages.append({"role": "user", "content": query})

        if self.model and self.device == "cuda" and self.tokenizer is not None:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_tokens = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2, # Low temperature for factual precision
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the generated response tokens
            response_tokens = outputs[0][prompt_tokens:]
            generated_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            total_tokens = prompt_tokens + len(response_tokens)
            return generated_text.strip(), total_tokens
        else:
            # Informative fallback output for CPU testing environment
            if context_chunks:
                top_context = context_chunks[0].get("parent_text", "")
                src = context_chunks[0].get("metadata", {}).get("source", "Document")
                pg = context_chunks[0].get("metadata", {}).get("page_number", 1)
                mock_text = (
                    f"**Context Retrieved from {src} (Page {pg}):**\n\n"
                    f"{top_context}\n\n"
                    f"*(Note: Running in CPU mode. Top {len(context_chunks)} document chunks retrieved and reranked.)*"
                )
            else:
                mock_text = f"No relevant enterprise document context found for: '{query}'."
            return mock_text, 256

    def ask(self, query: str, session_id: str = "default_session") -> Dict[str, Any]:
        """
        Main entry point for incoming user queries. Executes full RAG workflow.
        """
        start_time = time.time()

        # -------------------------------------------------------------
        # STEP 1: Check Redis Semantic & Exact Cache
        # -------------------------------------------------------------
        cached_response, cached_meta, match_type = self.cache.get(query)
        if cached_response and match_type in ["EXACT", "SEMANTIC"]:
            elapsed = time.time() - start_time
            # Update memory so conversation history stays coherent
            self.memory.add_message(session_id, "user", query)
            self.memory.add_message(session_id, "assistant", cached_response)

            return {
                "answer": cached_response,
                "retrieved_chunks": cached_meta.get("retrieved_chunks", []),
                "prompt_tokens": f"{cached_meta.get('tokens', 0)} tokens (CACHED)",
                "confidence_score": 1.0 if match_type == "EXACT" else 0.95,
                "cache_status": match_type,
                "latency_seconds": round(elapsed, 3),
            }

        # -------------------------------------------------------------
        # STEP 2: Vector Search (Top-15 Child Chunks)
        # -------------------------------------------------------------
        raw_hits = self.vector_store.search(query, top_k_children=15)

        # -------------------------------------------------------------
        # STEP 3: Cross-Encoder Reranking (Top-3 Parent Chunks)
        # -------------------------------------------------------------
        reranked_parents = self._rerank_chunks(query, raw_hits, top_n=3)

        # -------------------------------------------------------------
        # STEP 4: Retrieve Session Chat Memory
        # -------------------------------------------------------------
        chat_history = self.memory.get_history(session_id)

        # -------------------------------------------------------------
        # STEP 5: Generate Answer via Llama 3
        # -------------------------------------------------------------
        final_answer, total_tokens = self._generate_response(query, reranked_parents, chat_history)

        # calculate average confidence score from top reranked chunks
        avg_confidence = (
            sum(c.get("rerank_score", 0.0) for c in reranked_parents) / len(reranked_parents)
            if reranked_parents else 0.0
        )

        hud_chunk_payload = [
            {
                "rank": idx + 1,
                "source": c.get("metadata", {}).get("source", "Document"),
                "page": c.get("metadata", {}).get("page_number", 1),
                "rerank_score": round(c.get("rerank_score", 0.0), 4),
                "parent_id": c.get("parent_id", ""),
                "text": c.get("parent_text", "")
            }
            for idx, c in enumerate(reranked_parents)
        ]

        # -------------------------------------------------------------
        # STEP 6: Save Result to Cache & Update Chat Memory
        # -------------------------------------------------------------
        meta_to_cache = {
            "retrieved_chunks": hud_chunk_payload,
            "tokens": total_tokens,
            "avg_confidence": round(avg_confidence, 4)
        }
        self.cache.set(query, final_answer, meta_to_cache)

        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(session_id, "assistant", final_answer)

        elapsed = time.time() - start_time

        return {
            "answer": final_answer,
            "retrieved_chunks": hud_chunk_payload,
            "prompt_tokens": f"{total_tokens} tokens",
            "confidence_score": round(avg_confidence, 4),
            "cache_status": "MISS",
            "latency_seconds": round(elapsed, 3),
        }