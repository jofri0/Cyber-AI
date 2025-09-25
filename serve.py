from fastapi import FastAPI
from pydantic import BaseModel
from modules import programming, cybersecurity, general

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_ai(query: Query):
    q = query.question.lower()
    if 'hack' in q or 'security' in q:
        return {"answer": cybersecurity.answer(query.question)}
    elif 'code' in q or 'program' in q:
        return {"answer": programming.answer(query.question)}
    else:
        return {"answer": general.answer(query.question)}
