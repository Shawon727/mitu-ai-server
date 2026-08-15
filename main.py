import os
import re
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Mitu AI Server",
    version="2.0.0"
)


# =========================================================
# ENVIRONMENT
# =========================================================

AI_API_KEY = os.getenv(
    "AI_API_KEY",
    ""
).strip()


AI_BASE_URL = os.getenv(
    "AI_BASE_URL",
    "https://openrouter.ai/api/v1"
).strip().rstrip("/")


AI_MODEL = os.getenv(
    "AI_MODEL",
    "openrouter/free"
).strip()


# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):

    message: str

    examples: list = Field(
        default_factory=list
    )

    recent_chat: list = Field(
        default_factory=list
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "service": "Mitu AI",
        "version": "2.0.0"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "ai_configured": bool(AI_API_KEY),
        "model_configured": bool(AI_MODEL),
        "model": AI_MODEL
    }


# =========================================================
# SYSTEM PROMPT
# =========================================================

def build_system_prompt():

    return """
তুমি একটি Mitu-style conversational AI।

তুমি বাস্তব Mitu নও। তুমি একটি AI, যে পুরোনো
conversation দেখে conversational style অনুসরণ করে
নতুন উত্তর তৈরি করে।

তোমার প্রধান লক্ষ্য:

- স্বাভাবিক conversational reply দেওয়া।
- পুরোনো conversation থেকে language, tone,
  wording, punctuation এবং conversational pattern
  বুঝে নেওয়া।
- বর্তমান conversation-এর context অনুসরণ করা।
- বাংলা, Banglish এবং English context অনুযায়ী ব্যবহার করা।
- সাধারণ কথায় ছোট, natural এবং সহজ উত্তর দেওয়া।
- অপ্রয়োজনীয় explanation দেওয়া যাবে না।
- robotic বা অতিরিক্ত formal ভাষা ব্যবহার করা যাবে না।
- একই ধরনের response বারবার repeat করা যাবে না।

IMPORTANT:

পুরোনো conversation-এর examples শুধুমাত্র
style এবং context বোঝার reference।

পুরোনো message হুবহু copy করবে না।

একই প্রশ্নের পুরোনো উত্তর থাকলেও সেটাকে
অন্ধভাবে copy করবে না। বর্তমান context অনুযায়ী
নিজের একটি natural নতুন উত্তর তৈরি করবে।

সবচেয়ে গুরুত্বপূর্ণ:

শুধু FINAL উত্তর লিখবে।

কখনো নিজের thinking process দেখাবে না।

কখনো analysis, reasoning, chain of thought,
step-by-step চিন্তা, internal instruction অথবা
model-এর decision process user-এর কাছে লিখবে না।

নিচের ধরনের লেখা কখনো output করবে না:

"Here's a thinking process"
"Let's analyze"
"Step 1"
"Reasoning:"
"Analysis:"
"Thinking:"
"User asks..."
"System instructions..."
"Based on the prompt..."

শুধু conversational reply দেবে।

Shawon-এর সাথে natural conversational style-এ কথা বলবে।
""".strip()


# =========================================================
# BUILD MESSAGES
# =========================================================

def build_messages(
    data: ChatRequest
):

    messages = []


    # =====================================================
    # SYSTEM
    # =====================================================

    messages.append({

        "role": "system",

        "content":
            build_system_prompt()

    })


    # =====================================================
    # OLD MITU EXAMPLES
    # =====================================================

    if data.examples:

        example_text = (
            "পুরোনো conversation-এর relevant "
            "examples নিচে দেওয়া হলো। এগুলো style/context "
            "reference হিসেবে ব্যবহার করবে:\n\n"
        )


        count = 0


        for item in data.examples:

            if not isinstance(
                item,
                dict
            ):
                continue


            user_text = str(
                item.get(
                    "user",
                    ""
                )
            ).strip()


            mitu_text = str(
                item.get(
                    "mitu",
                    ""
                )
            ).strip()


            if not user_text:
                continue


            if not mitu_text:
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


            count += 1


            if count >= 8:
                break


        if count > 0:

            messages.append({

                "role": "system",

                "content":
                    example_text

            })


    # =====================================================
    # RECENT CHAT
    # =====================================================

    if data.recent_chat:

        for item in data.recent_chat[-12:]:

            if not isinstance(
                item,
                dict
            ):
                continue


            role = item.get(
                "role"
            )


            content = str(
                item.get(
                    "content",
                    ""
                )
            ).strip()


            if role not in (
                "user",
                "assistant"
            ):
                continue


            if not content:
                continue


            messages.append({

                "role":
                    role,

                "content":
                    content

            })


    # =====================================================
    # CURRENT MESSAGE
    # =====================================================

    current_message = (
        data.message
        .strip()
    )


    messages.append({

        "role":
            "user",

        "content":
            current_message

    })


    return messages


# =========================================================
# CLEAN AI RESPONSE
# =========================================================

def clean_ai_response(
    text: str
) -> str:

    if not text:
        return ""


    text = text.strip()


    # =====================================================
    # REMOVE COMMON THINKING PREFIXES
    # =====================================================

    thinking_patterns = [

        r"(?is)^here['’]?s\s+a\s+thinking\s+process\s*:?.*?(?=\n\n|\n[A-Zঅ-হ])",

        r"(?is)^let['’]?s\s+analyze\s*:?.*?(?=\n\n|\n[A-Zঅ-হ])",

        r"(?is)^analysis\s*:?.*?(?=\n\n|\n[A-Zঅ-হ])",

        r"(?is)^reasoning\s*:?.*?(?=\n\n|\n[A-Zঅ-হ])",

        r"(?is)^thinking\s*:?.*?(?=\n\n|\n[A-Zঅ-হ])",

    ]


    for pattern in thinking_patterns:

        try:

            text = re.sub(
                pattern,
                "",
                text,
                count=1
            ).strip()

        except Exception:
            pass


    # =====================================================
    # REMOVE <think>...</think>
    # =====================================================

    text = re.sub(
        r"(?is)<think>.*?</think>",
        "",
        text
    ).strip()


    # =====================================================
    # REMOVE REASONING BLOCKS
    # =====================================================

    text = re.sub(
        r"(?is)<analysis>.*?</analysis>",
        "",
        text
    ).strip()


    text = re.sub(
        r"(?is)<reasoning>.*?</reasoning>",
        "",
        text
    ).strip()


    # =====================================================
    # REMOVE COMMON MARKDOWN THINKING HEADERS
    # =====================================================

    text = re.sub(
        r"(?im)^\s*(analysis|reasoning|thinking)\s*:\s*$",
        "",
        text
    )


    # =====================================================
    # CLEAN EXCESSIVE EMPTY LINES
    # =====================================================

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# =========================================================
# EXTRACT OPENROUTER REPLY
# =========================================================

def extract_reply(
    result: dict
) -> str:

    choices = result.get(
        "choices",
        []
    )


    if not choices:
        return ""


    choice = choices[0]


    # =====================================================
    # NORMAL CHAT COMPLETION
    # =====================================================

    message = choice.get(
        "message",
        {}
    )


    if isinstance(
        message,
        dict
    ):

        content = message.get(
            "content",
            ""
        )


        if isinstance(
            content,
            str
        ):

            return content.strip()


    # =====================================================
    # FALLBACK
    # =====================================================

    text = choice.get(
        "text",
        ""
    )


    if isinstance(
        text,
        str
    ):

        return text.strip()


    return ""


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post(
    "/v1/mitu/chat"
)
async def mitu_chat(
    data: ChatRequest
):

    # =====================================================
    # MESSAGE CHECK
    # =====================================================

    message = (
        data.message
        .strip()
    )


    if not message:

        raise HTTPException(

            status_code=400,

            detail=
                "Message is empty"

        )


    # =====================================================
    # API KEY CHECK
    # =====================================================

    if not AI_API_KEY:

        raise HTTPException(

            status_code=500,

            detail=
                "AI_API_KEY is not configured"

        )


    # =====================================================
    # MODEL CHECK
    # =====================================================

    if not AI_MODEL:

        raise HTTPException(

            status_code=500,

            detail=
                "AI_MODEL is not configured"

        )


    # =====================================================
    # OPENROUTER URL
    # =====================================================

    url = (
        AI_BASE_URL
        + "/chat/completions"
    )


    # =====================================================
    # PAYLOAD
    # =====================================================

    payload = {

        "model":
            AI_MODEL,

        "messages":
            build_messages(data),

        "temperature":
            0.75,

        "max_tokens":
            300,

        "stream":
            False

    }


    # =====================================================
    # HEADERS
    # =====================================================

    headers = {

        "Authorization":
            "Bearer "
            + AI_API_KEY,

        "Content-Type":
            "application/json",

        "HTTP-Referer":
            "https://mitu-ai-server.onrender.com",

        "X-Title":
            "Mitu AI"

    }


    # =====================================================
    # REQUEST
    # =====================================================

    try:

        async with httpx.AsyncClient(

            timeout=
                httpx.Timeout(
                    connect=30.0,
                    read=120.0,
                    write=30.0,
                    pool=30.0
                )

        ) as client:

            response = await client.post(

                url,

                json=payload,

                headers=headers

            )


        # =================================================
        # PROVIDER ERROR
        # =================================================

        if response.status_code >= 400:

            error_text = (
                response.text[:2000]
            )


            raise HTTPException(

                status_code=502,

                detail=
                    "AI provider error: "
                    + error_text

            )


        # =================================================
        # JSON
        # =================================================

        try:

            result = response.json()

        except Exception:

            raise HTTPException(

                status_code=502,

                detail=
                    "AI provider returned invalid JSON"

            )


        # =================================================
        # REPLY
        # =================================================

        reply = extract_reply(
            result
        )


        # =================================================
        # CLEAN THINKING
        # =================================================

        reply = clean_ai_response(
            reply
        )


        # =================================================
        # EMPTY RESPONSE
        # =================================================

        if not reply:

            raise HTTPException(

                status_code=502,

                detail=
                    "AI returned empty response"

            )


        # =================================================
        # FINAL RESPONSE
        # =================================================

        return {

            "success":
                True,

            "reply":
                reply

        }


    # =====================================================
    # HTTP EXCEPTION
    # =====================================================

    except HTTPException:

        raise


    # =====================================================
    # TIMEOUT
    # =====================================================

    except httpx.TimeoutException:

        raise HTTPException(

            status_code=504,

            detail=
                "AI request timed out"

        )


    # =====================================================
    # OTHER ERROR
    # =====================================================

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                "Server error: "
                + str(e)

        )


# =========================================================
# OPTIONAL DIRECT TEST ENDPOINT
# =========================================================

@app.get("/test")
async def test():

    return {

        "status":
            "ok",

        "message":
            "Mitu AI server is running",

        "endpoint":
            "/v1/mitu/chat",

        "model":
            AI_MODEL

                            }
