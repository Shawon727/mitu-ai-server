import os
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Mitu AI Server",
    version="1.0.0"
)


# =========================================================
# ENVIRONMENT
# =========================================================

AI_API_KEY = os.getenv("AI_API_KEY", "")

AI_BASE_URL = os.getenv(
    "AI_BASE_URL",
    "https://openrouter.ai/api/v1"
)

AI_MODEL = os.getenv(
    "AI_MODEL",
    ""
)


# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):

    message: str

    examples: list = []

    recent_chat: list = []


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "service": "Mitu AI",
        "version": "1.0.0"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "ai_configured": bool(
            AI_API_KEY and AI_MODEL
        )
    }


# =========================================================
# SYSTEM PROMPT
# =========================================================

def build_system_prompt():

    return """
তুমি একটি Mitu-style conversational AI।

তোমার লক্ষ্য হলো একজন পুরোনো পরিচিত মানুষের
conversation style অনুসরণ করে স্বাভাবিকভাবে কথা বলা।

IMPORTANT:

- তুমি বাস্তব Mitu নও।
- নিজেকে বাস্তব ব্যক্তি হিসেবে দাবি করবে না।
- দেওয়া পুরোনো conversation থেকে ভাষা, tone,
  wording, punctuation এবং conversational pattern
  বুঝে উত্তর তৈরি করবে।
- পুরোনো message হুবহু copy করবে না যদি না প্রয়োজন হয়।
- বর্তমান কথার context অনুযায়ী নতুন উত্তর তৈরি করবে।
- অপ্রয়োজনীয় explanation দেবে না।
- robotic বা অতিরিক্ত formal ভাষা ব্যবহার করবে না।
- বাংলা, Banglish এবং English—যেটা conversation-এর
  context অনুযায়ী natural সেটাই ব্যবহার করবে।
- সাধারণ কথায় ছোট এবং natural উত্তর দেবে।
- user-এর প্রশ্ন অনুযায়ী উত্তর দেবে।
- একই ধরনের response বারবার repeat করবে না।
- conversation context ভুলে যাবে না।
- পুরোনো examples-কে facts হিসেবে অন্ধভাবে অনুসরণ করবে না;
  সেগুলো মূলত conversational style শেখার reference।

Shawon-এর সাথে natural conversational style-এ কথা বলবে।
""".strip()


# =========================================================
# BUILD PROMPT
# =========================================================

def build_messages(data: ChatRequest):

    messages = []

    messages.append({
        "role": "system",
        "content": build_system_prompt()
    })


    # -----------------------------------------------------
    # OLD MITU EXAMPLES
    # -----------------------------------------------------

    if data.examples:

        example_text = (
            "পুরোনো conversation-এর relevant examples:\n\n"
        )

        for item in data.examples[:8]:

            if not isinstance(item, dict):
                continue

            user_text = str(
                item.get("user", "")
            ).strip()

            mitu_text = str(
                item.get("mitu", "")
            ).strip()

            if not user_text or not mitu_text:
                continue

            example_text += (
                "Shawon: "
                + user_text
                + "\n"
            )

            example_text += (
                "Mitu: "
                + mitu_text
                + "\n\n"
            )


        if len(example_text) > 100:

            messages.append({
                "role": "system",
                "content": example_text
            })


    # -----------------------------------------------------
    # RECENT CHAT
    # -----------------------------------------------------

    if data.recent_chat:

        for item in data.recent_chat[-12:]:

            if not isinstance(item, dict):
                continue

            role = item.get("role")

            content = str(
                item.get("content", "")
            ).strip()

            if role not in (
                "user",
                "assistant"
            ):
                continue

            if not content:
                continue

            messages.append({
                "role": role,
                "content": content
            })


    # -----------------------------------------------------
    # CURRENT MESSAGE
    # -----------------------------------------------------

    messages.append({
        "role": "user",
        "content": data.message.strip()
    })


    return messages


# =========================================================
# CHAT
# =========================================================

@app.post("/v1/mitu/chat")
async def mitu_chat(data: ChatRequest):

    if not data.message.strip():

        raise HTTPException(
            status_code=400,
            detail="Message is empty"
        )


    if not AI_API_KEY:

        raise HTTPException(
            status_code=500,
            detail="AI_API_KEY is not configured"
        )


    if not AI_MODEL:

        raise HTTPException(
            status_code=500,
            detail="AI_MODEL is not configured"
        )


    url = (
        AI_BASE_URL.rstrip("/")
        + "/chat/completions"
    )


    payload = {

        "model": AI_MODEL,

        "messages":
            build_messages(data),

        "temperature":
            0.75,

        "max_tokens":
            300,

        "stream":
            False
    }


    headers = {

        "Authorization":
            "Bearer " + AI_API_KEY,

        "Content-Type":
            "application/json",

        "HTTP-Referer":
            "https://mitu-ai.onrender.com",

        "X-Title":
            "Mitu AI"
    }


    try:

        async with httpx.AsyncClient(
            timeout=120.0
        ) as client:

            response = await client.post(
                url,
                json=payload,
                headers=headers
            )


        if response.status_code >= 400:

            raise HTTPException(
                status_code=502,
                detail=(
                    "AI provider error: "
                    + response.text[:1000]
                )
            )


        result = response.json()


        choices = result.get(
            "choices",
            []
        )


        if not choices:

            raise HTTPException(
                status_code=502,
                detail="AI returned no choices"
            )


        message = choices[0].get(
            "message",
            {}
        )


        reply = str(
            message.get(
                "content",
                ""
            )
        ).strip()


        if not reply:

            raise HTTPException(
                status_code=502,
                detail="AI returned empty response"
            )


        return {

            "success":
                True,

            "reply":
                reply
        }


    except httpx.TimeoutException:

        raise HTTPException(
            status_code=504,
            detail="AI request timed out"
        )


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
  )
