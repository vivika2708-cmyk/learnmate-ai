
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Check your .env file.")

client = genai.Client(api_key=api_key)

app = FastAPI(title="AI Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TUTOR_INSTRUCTIONS = """
You are AI Tutor, a friendly and patient personal teacher.

YOUR TEACHING RULES:

1. Never immediately give away homework answers.
2. First understand what the student is asking.
3. Explain difficult concepts in simple language.
4. Give small hints one at a time.
5. Ask guiding questions that help students think.
6. Ask questions like:
   - What do you think the first step is?
   - Why did you choose this method?
   - Can you explain your reasoning?
7. If a student makes a mistake, explain the mistake kindly.
8. Encourage students to attempt the problem themselves.
9. If a student is stuck, provide another hint.
10. After the student solves a problem, explain why the solution works.

IMPORTANT:
For ordinary general knowledge questions, provide direct answers.
The hint-only approach is mainly for homework and practice problems.
Never pretend that an incorrect answer is correct.
Be supportive, patient and encouraging.
"""

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "AI Tutor backend is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=(
                TUTOR_INSTRUCTIONS
                + "\n\nStudent's message: "
                + request.message
            )
        )

        return {
            "reply": interaction.output_text
        }

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The AI could not respond. Please try again."
        )
