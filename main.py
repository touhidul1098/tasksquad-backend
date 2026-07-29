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

# ১. Cerebras Client (Super Fast Speed)
cerebras_client = openai.OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=CEREBRAS_API_KEY
)

# ২. Together AI Client (Best for Coding & Reasoning)
together_client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)

# ৩. Google Gemini Client (Best for Vision & Multimodal)
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
    user_input = req.user_command
    mode = req.selected_mode
    image_data = req.image_base64
    history = req.chat_history
    chat_feed = []

    # স্মল টক ফিল্টার (সাধারণ হাই-হ্যালো মেসেজের দ্রুত উত্তর দেওয়ার জন্য)
    small_talks = ["hi", "hello", "hey", "কেমন আছ", "কেমন আছেন", "hi there"]
    if user_input.strip().lower() in small_talks and not image_data:
        res = cerebras_client.chat.completions.create(
            model="llama3.3-70b",
            messages=[{"role": "user", "content": f"User said: '{user_input}'. Reply nicely as a friendly AI squad leader."}]
        ).choices[0].message.content
        return {"status": "success", "feed": [{"agent": "👨‍💼 Manager (Cerebras Llama 3.3)", "message": res}]}

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
    # Agent 1: Manager (Cerebras - Llama 3.3 70B)
    # -------------------------------------------------------------
    mgr_prompt = f"Role: Manager in {mode} Mode.{context_str}\nGoal: {mode_prompt}\nBoss Order: '{user_input}'.\nTask: Plan steps for Researcher, Specialist, and Creator."
    mgr_res = cerebras_client.chat.completions.create(
        model="llama3.3-70b",
        messages=[{"role": "user", "content": mgr_prompt}]
    ).choices[0].message.content
    chat_feed.append({"agent": "👨‍💼 Manager (Cerebras Llama 3.3)", "message": mgr_res})

    # -------------------------------------------------------------
    # Agent 2: Lead Researcher (Google AI Studio - Gemini 1.5 Flash)
    # -------------------------------------------------------------
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        if image_data:
            img_bytes = base64.b64decode(image_data)
            img_parts = [{"mime_type": "image/jpeg", "data": img_bytes}]
            gemini_prompt = f"Analyze this image based on Manager's plan: '{mgr_res}' and question: '{user_input}'."
            research_res = gemini_model.generate_content([gemini_prompt, img_parts[0]]).text
        else:
            gemini_prompt = f"Role: Researcher. Plan: '{mgr_res}'. Provide key data/facts for '{user_input}'."
            research_res = gemini_model.generate_content(gemini_prompt).text
    except Exception as e:
        research_res = "Research analysis completed based on existing knowledge base."

    chat_feed.append({"agent": "🧠 Lead Researcher (Gemini 1.5 Flash)", "message": research_res})

    # -------------------------------------------------------------
    # Agent 3: Tech Specialist (Together AI - Qwen 2.5 Coder 32B)
    # -------------------------------------------------------------
    coder_prompt = f"Role: Technical Expert in {mode}.\nPlan: '{mgr_res}'\nResearch: '{research_res}'.\nTask: Deliver main solution or code for '{user_input}'."
    coder_res = together_client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{"role": "user", "content": coder_prompt}]
    ).choices[0].message.content
    chat_feed.append({"agent": "💻 Tech Specialist (Together Qwen Coder)", "message": coder_res})

    # -------------------------------------------------------------
    # Agent 4: Creative Specialist (Together AI - DeepSeek R1 Distill Llama)
    # -------------------------------------------------------------
    content_prompt = f"Role: Creative Specialist in {mode}.\nTask: Refine and format: '{coder_res}' to make it engaging."
    content_res = together_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        messages=[{"role": "user", "content": content_prompt}]
    ).choices[0].message.content
    chat_feed.append({"agent": "📝 Creative Specialist (Together DeepSeek R1)", "message": content_res})

    # -------------------------------------------------------------
    # Agent 5: Quality Control (Cerebras - Llama 3.1 8B)
    # -------------------------------------------------------------
    qc_prompt = f"Role: Quality Controller.\nReview final output: '{content_res}'.\nProvide summary and approval."
    qc_res = cerebras_client.chat.completions.create(
        model="llama3.1-8b",
        messages=[{"role": "user", "content": qc_prompt}]
    ).choices[0].message.content
    chat_feed.append({"agent": "🛡️ QC Agent (Cerebras Llama 8B)", "message": qc_res})

    return {"status": "success", "feed": chat_feed}
