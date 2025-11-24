from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
import redis
import json

from retrieval import Retriever
from generation import QAModel
from mlm import MLMModel
from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

app = FastAPI(
    title="Talk-to-PDF API",
    description="RAG + MLM + Multi-turn Chat on PDF chunks",
    version="1.2"
)

class MLMRequest(BaseModel):
    text: str 

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    top_k: int = 1

retriever = Retriever()
qa_model = QAModel()
mlm_model = MLMModel()

@app.get("/")
def root():
    return {"message": "Talk-to-PDF API is running."}

@app.post("/mlm")
def predict_mask(req: MLMRequest):
    if "[MASK]" not in req.text:
        raise HTTPException(status_code=400, detail="Text must contain a [MASK] token.")
    predicted_word = mlm_model.predict_mask(req.text)
    return {
        "input_text": req.text,
        "predicted_word": predicted_word,
        "filled_text": req.text.replace("[MASK]", predicted_word)
    }

@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Retrieve conversation history from Redis. History is stored as a List of JSON strings.
    # r.lrange(key, 0, -1) retrieves all elements in the List.
    try:
        redis_history_items = r.lrange(session_id, 0, -1)
        # Parse each item in the list from a JSON string to a Python dictionary
        prev_history = [json.loads(item) for item in redis_history_items]
    except redis.exceptions.ResponseError as e:
        print(f"Redis type error during history retrieval for {session_id}: {e}")
        prev_history = []


    # Answer question using hybrid QA model (BERT + Ollama)
    answer = qa_model.answer_question(query=req.query, session_id=session_id, top_k=req.top_k)

    # Append current turn to the Python list
    current_turn = {"user": req.query, "system": answer}
    prev_history.append(current_turn)

    # Store the latest turn as a new element at the end of the Redis List.
    # The current turn object is serialized to a JSON string before being pushed.
    r.rpush(session_id, json.dumps(current_turn))

    return {
        "session_id": session_id,
        "answer": answer,
        "history": prev_history
    }