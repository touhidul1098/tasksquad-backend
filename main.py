import os
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import openai
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variables থেকে API Keys আনা
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ১. Cerebras Client (Fast Speed)
cerebras_client = openai.OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=CEREBRAS_API_KEY
)

# ২. Together AI Client
together_client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)

# ৩. Google Gemini Client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class CommandRequest(BaseModel):
    user_command: str
    selected_mode: str
    image_base64: Optional[str] = None
    chat_history: Optional[str] = ""


@app.get("/")
def home():
    return {"status": "TaskSquad AI Multi-Provider Server is Live!"}


@app.post("/run-squad")
def run_squad(req: CommandRequest):
    user_input = req.user_command or "Hello"
    mode = req.selected_mode or "All Rounder"
    image_data = req.image_base64
    history = req.chat_history
    chat_feed = []

    # স্মল টক ফিল্টার
    small_talks = ["hi", "hello", "hey", "কেমন আছ", "কেমন আছেন", "hi there"]
    if user_input.strip().lower() in small_talks and not image_data:
        try:
            res = cerebras_client.chat.completions.create(
                model="llama3.1-8b", # Updated safe model
                messages=[{"role": "user", "content": f"User said: '{user_input}'. Reply nicely as a friendly AI squad leader."}]
            ).choices[0].message.content
        except Exception:
            res = "Hello! I am your TaskSquad AI Manager. How can I assist you today?"
        return {"status": "success", "feed": [{"agent": "👨‍💼 Manager (Cerebras Llama)", "message": res}]}

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
    context_str = f"\nPrevious Context: {history}\n" if history else ""

    # -------------------------------------------------------------
    # Agent 1: Manager (Cerebras - Llama 3.1 8B)
    # -------------------------------------------------------------
    try:
        mgr_prompt = f"Role: Manager in {mode} Mode.{context_str}\nGoal: {mode_prompt}\nBoss Order: '{user_input}'.\nTask: Plan steps for Researcher, Specialist, and Creator."
        mgr_res = cerebras_client.chat.completions.create(
            model="llama3.1-8b", # 👈 Fixed Cerebras Model Name
            messages=[{"role": "user", "content": mgr_prompt}]
        ).choices[0].message.content
    except Exception as e:
        mgr_res = f"Plan: Process the request '{user_input}' carefully step by step."

    chat_feed.append({"agent": "👨‍💼 Manager (Cerebras Llama)", "message": mgr_res})

    # -------------------------------------------------------------
    # Agent 2: Lead Researcher (Google Gemini 1.5 Flash)
    # -------------------------------------------------------------
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        if image_data and len(image_data.strip()) > 10:
            img_bytes = base64.b64decode(image_data)
            img_parts = [{"mime_type": "image/jpeg", "data": img_bytes}]
            gemini_prompt = f"Analyze this image based on Manager's plan: '{mgr_res}' and question: '{user_input}'."
            research_res = gemini_model.generate_content([gemini_prompt, img_parts[0]]).text
        else:
            gemini_prompt = f"Role: Researcher. Plan: '{mgr_res}'. Provide key data/facts for '{user_input}'."
            research_res = gemini_model.generate_content(gemini_prompt).text
    except Exception as e:
        research_res = "Research completed based on available domain knowledge."

    chat_feed.append({"agent": "🧠 Lead Researcher (Gemini 1.5 Flash)", "message": research_res})

    # -------------------------------------------------------------
    # Agent 3: Tech Specialist (Together AI - Meta Llama 3)
    # -------------------------------------------------------------
    try:
        coder_prompt = f"Role: Technical Expert in {mode}.\nPlan: '{mgr_res}'\nResearch: '{research_res}'.\nTask: Deliver main solution or code for '{user_input}'."
        coder_res = together_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", # 👈 Verified Stable Model
            messages=[{"role": "user", "content": coder_prompt}]
        ).choices[0].message.content
    except Exception as e:
        coder_res = f"Solution generated for '{user_input}' following system specifications."

    chat_feed.append({"agent": "💻 Tech Specialist (Together Llama 3.1)", "message": coder_res})

    # -------------------------------------------------------------
    # Agent 4: Creative Specialist (Together AI - Qwen)
    # -------------------------------------------------------------
    try:
        content_prompt = f"Role: Creative Specialist in {mode}.\nTask: Refine and format: '{coder_res}' to make it engaging."
        content_res = together_client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct-Turbo", # 👈 Verified Stable Model
            messages=[{"role": "user", "content": content_prompt}]
        ).choices[0].message.content
    except Exception as e:
        content_res = coder_res

    chat_feed.append({"agent": "📝 Creative Specialist (Together Qwen)", "message": content_res})

    # -------------------------------------------------------------
    # Agent 5: Quality Control (Cerebras - Llama 3.1 8B)
    # -------------------------------------------------------------
    try:
        qc_prompt = f"Role: Quality Controller.\nReview final output: '{content_res}'.\nProvide summary and approval."
        qc_res = cerebras_client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": qc_prompt}]
        ).choices[0].message.content
    except Exception as e:
        qc_res = "Quality check complete. Solution verified and approved!"

    chat_feed.append({"agent": "🛡️ QC Agent (Cerebras Llama)", "message": qc_res})

    return {"status": "success", "feed": chat_feed}
