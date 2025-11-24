from config import TOKENIZER, QA_MODEL, CONFIG
import torch
import redis
from retrieval.retriever import Retriever
import ollama

class QAModel:
    """Hybrid Extractive + Generative QA with Redis-based chat context using BERT + Ollama."""

    def __init__(self, redis_enabled=True, ollama_model=CONFIG.get("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")):
        # BERT QA
        self.tokenizer = TOKENIZER
        self.model = QA_MODEL
        self.retriever = Retriever()

        # Redis setup
        self.redis_enabled = redis_enabled
        if self.redis_enabled:
            # Use .get with a default for safety, but rely on CONFIG for structure
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
            self.redis_client.rpush(session_id, f"You: {user_input}")
            self.redis_client.rpush(session_id, f"System: {answer}")

    def _get_conversation_context(self, session_id: str, max_turns: int = 5) -> str:
        if self.redis_enabled:
            history = self.redis_client.lrange(session_id, -max_turns * 2, -1)
            return "\n".join(history) if history else ""
        return ""

    # BertQA + Ollama hybrid
    def answer_question(self, query: str, session_id: str = None, top_k: int = 5) -> str:
        top_chunks = self.retriever.retrieve(query, top_k=top_k)
        
        MAX_BERT_TOKENS = 512 

        query_tokens_count = len(self.tokenizer.tokenize(query))
        
        # We reserve 4 tokens for special separators: [CLS], [SEP] after query, and two [SEP] markers potentially used by the tokenizer for the context end.
        RESERVED_TOKENS = 4 
        max_context_length = MAX_BERT_TOKENS - query_tokens_count - RESERVED_TOKENS 

        concatenated_context = ""
        current_token_count = 0
        
        # Iterate through all retrieved chunks and add them until the token limit is hit
        for chunk in top_chunks:
            chunk_tokens = self.tokenizer.tokenize(chunk)
            
            if current_token_count + len(chunk_tokens) + 1 <= max_context_length:
                concatenated_context += chunk + " "
                current_token_count += len(chunk_tokens) + 1
            else:
                break 
                
        context = concatenated_context.strip() 
        if not context:
            context = "" 
        

        if session_id:
            convo_context = self._get_conversation_context(session_id)
            if convo_context:
                context = convo_context + "\n" + context

        inputs = self.tokenizer.encode_plus(query, context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extractive prediction
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
        bert_answer = self.tokenizer.decode(answer_tokens, clean_up_tokenization_spaces=True)

        # Ollama LLM for fluent answer
        llm_prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

BERT Extractive Answer:
{bert_answer}

Question:
{query}

Answer:"""

        # Extract the response content from the nested dictionary structure.
        try:
            llm_response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": llm_prompt 
                    }
                ]
            )
            final_answer = llm_response['message']['content'].strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            final_answer = f"Could not generate a fluent answer. BERT's extractive answer was: {bert_answer}"

        if session_id:
            self._save_conversation(session_id, query, final_answer)

        return final_answer
