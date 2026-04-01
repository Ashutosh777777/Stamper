from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import Memory as mem
import scraper

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

responder = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

class StartRequest(BaseModel):
    name: str
    hobbies: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    hobbies: str

@app.post("/start")
def start_session(req: StartRequest):
    session_id = mem.start_session(req.name, req.hobbies)
    return {"session_id": session_id, "message": f"Session started for {req.name}"}

@app.post("/chat")
def chat(req: ChatRequest):
    mem.log("user", req.message)

    web_context = ""
    if scraper.should_search(req.message):
        web_context = scraper.search(req.message)

    long_term  = mem.retrieve_context(req.message)
    short_term = mem.get_recent_turns(n=10)

    response = responder.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": (
                f"You are Natasha, a personal assistant and good friend. "
                f"The user's hobbies are {req.hobbies}. Keep responses in plain speakable text. "
                f"Be warm, occasionally witty, and adapt your tone to the user.\n\n"
                + long_term
                + (f"\n\nWeb context:\n{web_context}" if web_context else "")
            )},
            *short_term,
            {"role": "user", "content": req.message}
        ]
    )

    reply = response.choices[0].message.content
    mem.log("assistant", reply)
    return {"reply": reply}

@app.post("/end")
def end_session():
    mem.end_session()
    return {"message": "Session ended and memory saved."}

@app.get("/history")
def get_history():
    return {"history": mem.get_recent_turns(n=20)}