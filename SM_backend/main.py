# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import asyncio

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5500")  # adjust if needed

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY required in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Show Me Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://127.0.0.1:5500"],  # allow frontend dev addresses
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ItineraryRequest(BaseModel):
    city: str
    days: int
    interests: list[str]

class ItineraryResponse(BaseModel):
    text: str

@app.post("/generate-itinerary", response_model=ItineraryResponse)
async def generate_itinerary(req: ItineraryRequest):
    # Basic validation
    if not req.city or req.days < 1 or not req.interests:
        raise HTTPException(status_code=400, detail="city, days (>=1) and at least one interest required")

    # Build instructive prompt — ask for JSON-like or plain text itinerary
    prompt = f"""
You are a friendly travel planner. Create a detailed {req.days}-day itinerary for the city "{req.city}".
User interests: {', '.join(req.interests)}.
Return a clear, day-by-day plan with short descriptions and a few concrete suggestions (1-2 places per half-day).
Keep language concise and helpful.
Label days as "Day 1", "Day 2", etc.
"""

    try:
        # Use the Responses API (or Chat completions if you prefer). This uses OpenAI's python client.
        # The client.responses.create interface returns structured output; here we use simple text output.
        resp = await asyncio.to_thread(lambda: client.responses.create(
            model="gpt-4o",        # change to a model you have access to (e.g., "gpt-4o-mini" or "gpt-4o")
            input=prompt,
            temperature=0.7,
            max_output_tokens=800
        ))

        # `resp.output_text` contains the combined text
        text = getattr(resp, "output_text", None)
        if not text:
            # fallback parse from 'resp.output' if necessary
            text = str(resp)

        return ItineraryResponse(text=text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")
