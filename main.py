import os
import base64
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import openai
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

cerebras_client = openai.OpenAI(base_url="https://api.cerebras.ai/v1", api_key=CEREBRAS_API_KEY)
together_client = openai.OpenAI(base_url="https://api.together.xyz/v1", api_key=TOGETHER_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

class CommandRequest(BaseModel):
    user_command: str
    selected_mode: str
    image_base64: Optional[str] = None
    chat_history: Optional[str] = ""
    active_agents: Optional[List[str]] = None  # Dynamic Agent selection list

@app.get("/")
def home():
    return {"status": "TaskSquad AI Multi-Provider Server is Live!"}

@app.post("/run-squad")
def run_squad(req: CommandRequest):
    user_input = req.user_command or "Hello"
    mode = req.selected_mode or "All Rounder"
    image_data = req.image_base64
    history = req.chat_history or ""
    enabled_agents = req.active_agents or ["manager", "researcher", "tech", "creative", "qc"]
    chat_feed = []

    system_instructions = {
        "Writing": "Focus on high-quality creative and professional writing.",
        "Study": "Explain concepts simply with examples like an expert teacher.",
        "Design": "Focus on UI/UX layout, aesthetics, and visual design prompts.",
        "Coding": "Focus on highly optimized, error-free code and software architecture.",
        "All Rounder": "Provide versatile and comprehensive solutions.",
        "Business Strategy": "Focus on business execution, growth strategy, and ROI.",
        "Marketing": "Focus on viral marketing, engagement, and content strategies.",
        "Research": "Provide deep-dive analysis, facts, and well-researched insights.",
        "Prompt Generator": "Generate detailed prompts for Midjourney, Flux, or DALL-E."
    }

    mode_prompt = system_instructions.get(mode, "Handle this efficiently.")
    context_str = f"\nPrevious History: {history}\n" if history else ""

    mgr_res = "User request direct execution."
    research_res = "Direct domain analysis."
    coder_res = "Execution output ready."

    # 1. Manager
    if "manager" in enabled_agents:
        try:
            mgr_prompt = f"Role: Manager in {mode} Mode.{context_str}\nGoal: {mode_prompt}\nUser Request: '{user_input}'. Create plan."
            mgr_res = cerebras_client.chat.completions.create(
                model="llama-3.3-70b",
                messages=[{"role": "user", "content": mgr_prompt}]
            ).choices[0].message.content
            chat_feed.append({"agent": "👨‍💼 Manager (Cerebras Llama 3.3)", "message": mgr_res})
        except Exception as e:
            logging.error(f"Manager Error: {e}")

    # 2. Researcher
    if "researcher" in enabled_agents:
        try:
            if image_data and len(image_data.strip()) > 10 and gemini_client:
                clean_image_data = image_data.split(",")[1] if "," in image_data else image_data
                img_bytes = base64.b64decode(clean_image_data)
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'), f"Analyze image: {user_input}"]
                )
                research_res = response.text
            elif gemini_client:
                gemini_prompt = f"Role: Researcher. Plan: '{mgr_res}'. Request: '{user_input}'."
                response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=gemini_prompt)
                research_res = response.text
            chat_feed.append({"agent": "🧠 Lead Researcher (Gemini 2.5)", "message": research_res})
        except Exception as e:
            logging.error(f"Gemini Error: {e}")

    # 3. Tech Specialist
    if "tech" in enabled_agents:
        try:
            coder_prompt = f"Role: Tech Expert in {mode}. Plan: '{mgr_res}'. Research: '{research_res}'. Request: '{user_input}'."
            coder_res = together_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": coder_prompt}]
            ).choices[0].message.content
            chat_feed.append({"agent": "💻 Tech Specialist (Together Llama 3.1)", "message": coder_res})
        except Exception as e:
            logging.error(f"Tech Error: {e}")

    # 4. Creative Specialist
    if "creative" in enabled_agents:
        try:
            content_prompt = f"Role: Creative Specialist in {mode}. Refine: '{coder_res}'."
            content_res = together_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct-Turbo",
                messages=[{"role": "user", "content": content_prompt}]
            ).choices[0].message.content
            chat_feed.append({"agent": "📝 Creative Specialist (Together Qwen 72B)", "message": content_res})
        except Exception as e:
            logging.error(f"Creative Error: {e}")

    # 5. Quality Control
    if "qc" in enabled_agents:
        try:
            qc_prompt = f"Role: QC Agent. Review output of request: '{user_input}'. Provide final approved answer."
            qc_res = cerebras_client.chat.completions.create(
                model="llama-3.3-70b",
                messages=[{"role": "user", "content": qc_prompt}]
            ).choices[0].message.content
            chat_feed.append({"agent": "🛡️ QC Agent (Cerebras Llama 3.3)", "message": qc_res})
        except Exception as e:
            logging.error(f"QC Error: {e}")

    return {"status": "success", "feed": chat_feed}
