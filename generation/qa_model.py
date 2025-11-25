from config import TOKENIZER, CONFIG
import torch
import redis
from retrieval.retriever import Retriever
import ollama
from typing import List
import json

class QAModel:
    """Generative QA with Redis-based chat context using Ollama."""

    def __init__(self, redis_enabled=True, ollama_model=CONFIG.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")):
        self.tokenizer = TOKENIZER
        self.retriever = Retriever()

        # Redis setup
        self.redis_enabled = redis_enabled
        if self.redis_enabled:
            self.redis_client = redis.Redis(
                host=CONFIG.get("REDIS_HOST", "localhost"),
                port=CONFIG.get("REDIS_PORT", 6379),
                password=CONFIG.get("REDIS_PASSWORD", None),
                decode_responses=True
            )

        # Ollama LLM
        self.ollama_model = ollama_model

    # Redis chat utilities
    def _save_conversation(self, session_id: str, user_input: str, answer: str):
        if self.redis_enabled:
            # Store as JSON objects for safe multi-turn retrieval
            self.redis_client.rpush(session_id, json.dumps({"role": "user", "content": user_input}))
            self.redis_client.rpush(session_id, json.dumps({"role": "system", "content": answer}))

    def _get_conversation_context(self, session_id: str, max_turns: int = 5) -> str:
        if self.redis_enabled:
            # Get last max_turns*2 messages as JSON
            history_items = self.redis_client.lrange(session_id, -max_turns*2, -1)
            context_texts = []
            for item in history_items:
                try:
                    obj = json.loads(item)
                    context_texts.append(f"{obj['role'].capitalize()}: {obj['content']}")
                except (json.JSONDecodeError, KeyError):
                    # fallback for any legacy plain-text entries
                    context_texts.append(str(item))
            return "\n".join(context_texts) if context_texts else ""
        return ""

    # Helpers
    def _ensure_list(self, x) -> List[str]:
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            if len(x) > 0 and isinstance(x[0], list):
                return x[0]
            return list(x)
        return [x]

    # Main QA function
    def answer_question(self, query: str, session_id: str = None, top_k: int = 5) -> str:
        # 1) Retrieve top_k chunks (normalize return)
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        top_chunks = self._ensure_list(retrieved)

        concatenated = "\n\n".join(top_chunks[:top_k])

        convo_context = ""
        if session_id:
            convo_context = self._get_conversation_context(session_id)
            if convo_context:
                concatenated = convo_context + "\n\n" + concatenated

        llm_prompt = (
            "You are a helpful assistant. Answer ONLY using the information provided below. "
            "If the answer is not in the context, say: 'The document does not provide this information.'\n\n"
            f"Context:\n{concatenated}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"

)
        try:
            llm_response = ollama.chat(
                messages=[{"role": "user", "content": llm_prompt}],
                model=self.ollama_model
            )
            
            final_answer = ""
            if isinstance(llm_response, dict):
                # Dictionary response: {'message': {'content': '...'}}
                final_answer = llm_response.get("message", {}).get("content", "")
            elif hasattr(llm_response, 'message') and hasattr(llm_response.message, 'content'):
                final_answer = llm_response.message.content
            elif hasattr(llm_response, 'content'):
                final_answer = llm_response.content
            else:
                final_answer = f"Could not parse LLM response"
            
            final_answer = final_answer.strip() if final_answer else ""
            
        except Exception as e:
            print(f"Ollama error: {e}")
            final_answer = f"Could not generate a fluent answer."
        if session_id:
            self._save_conversation(session_id, query, final_answer)

        return final_answer
