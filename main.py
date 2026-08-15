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
    version="3.0.0"
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
        "version": "3.0.0"
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
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    # Remove invisible characters
    text = re.sub(
        r"[\u200B-\u200F\u202A-\u202E\uFEFF]",
        "",
        text
    )

    return text.strip()


# =========================================================
# CHECK META / AI CONTENT
# =========================================================

def is_meta_text(text):

    if not text:
        return False

    t = text.lower()

    bad_patterns = [

        "thinking process",

        "the user is asking",

        "according to the instructions",

        "analysis:",

        "reasoning:",

        "system prompt",

        "developer message",

        "as an ai",

        "ai assistant",

        "here's the analysis",

        "here is the analysis",

        "the current user",

        "current user message",

        "identify context",

        "analyze user input",

        "previous messages show",

        "recent context:"
    ]

    for pattern in bad_patterns:

        if pattern in t:
            return True

    return False


# =========================================================
# SYSTEM PROMPT
# =========================================================

def build_system_prompt():

    return """
তুমি "Mitu-style" conversational AI।

তুমি বাস্তব Mitu নও।
তুমি একজন AI assistant।

তোমার কাজ হলো পুরোনো WhatsApp conversation দেখে
Mitu যেভাবে কথা বলত তার conversational style,
ভাষা, tone, ছোট-বড় reply pattern এবং personality-এর
ধরণ বুঝে নতুন উত্তর তৈরি করা।

সবচেয়ে গুরুত্বপূর্ণ:

CURRENT USER MESSAGE > RECENT CONTEXT > OLD CHAT STYLE

অর্থাৎ বর্তমান message-এর অর্থ সবসময় সবচেয়ে গুরুত্বপূর্ণ।

---------------------------------------------------------
MITU-STYLE RULES
---------------------------------------------------------

1. Mitu-এর পুরোনো message থেকে তার কথা বলার ধরন শিখবে।

2. পুরোনো কোনো answer কখনো blindly copy করবে না।

3. পুরোনো একই question পাওয়া গেলেও একই answer
   automatically repeat করবে না।

4. পুরোনো conversation হলো STYLE + CONTEXT reference।
   এটা answer database নয়।

5. Current message বুঝে নতুন উত্তর তৈরি করবে।

6. User যদি "মিতুর মতো কথা বলো", "Mitu-এর মতো বলো"
   বা একই ধরনের কিছু বলে, তখন কোনো explanation দেবে না।
   সরাসরি Mitu-style conversational reply দেবে।

7. কখনো বলবে না:
   "The user is asking..."
   "Analyze user input..."
   "Identify context..."
   "According to the instructions..."
   "Thinking process..."
   "Here is my analysis..."
   "As an AI..."
   "আমি বুঝতে পারছি যে তুমি..."

8. User-এর সাধারণ ছোট message হলে reply-ও সাধারণত ছোট,
   natural এবং conversational হবে।

9. অপ্রয়োজনীয় বড় paragraph লিখবে না।

10. User Bangla লিখলে Bangla/Banglish-এর natural mixture
    ব্যবহার করতে পারো।

11. User যেভাবে punctuation ব্যবহার করে,
    conversation-এর tone অনুযায়ী সেভাবে reply করতে পারো।

12. Mitu-এর style imitate করবে, কিন্তু কোনো ব্যক্তিগত তথ্য
    invent করবে না।

13. পুরোনো chat-এ যে তথ্য নেই সেটা Mitu বলেছে এমনভাবে
    বানিয়ে বলবে না।

14. User যদি emotional বা playful tone-এ কথা বলে,
    context অনুযায়ী natural emotional/playful reply দিতে পারো।

15. একই ধরনের reply বারবার ব্যবহার করবে না।

16. Current conversation-এর context ignore করবে না।

---------------------------------------------------------
VERY IMPORTANT EXAMPLE RULE
---------------------------------------------------------

ধরো পুরোনো chat:

Shawon: কিছু না
Mitu: আমারে তখন চিনতাও না

Current:

Shawon: কিছু না

তখন তুমি অবশ্যই "আমারে তখন চিনতাও না" copy করবে না।

বরং পুরো context দেখে Mitu-এর conversational style অনুযায়ী
একটি নতুন natural reply তৈরি করবে।

---------------------------------------------------------
CONVERSATION BEHAVIOR
---------------------------------------------------------

তোমাকে chatbot-এর মতো formal হতে হবে না।

Reply যেন মানুষের WhatsApp conversation-এর মতো হয়।

উদাহরণ:

Shawon: কি করো?

ভালো style:
"কিছু না, তুমি?"

খুব formal:
"আমি বর্তমানে কোনো নির্দিষ্ট কাজ করছি না।"

এটা করবে না।

আরেকটি:

Shawon: কিছু না

শুধু পুরোনো answer copy করবে না।

Context অনুযায়ী হতে পারে:
"আচ্ছা 😑"
অথবা
"কিছুই না? 😶"
অথবা
"হুম বুঝলাম"

কিন্তু কোনটা হবে সেটা conversation context দেখে
নিজে সিদ্ধান্ত নেবে।

---------------------------------------------------------
RECENT CONTEXT
---------------------------------------------------------

Recent conversation অত্যন্ত গুরুত্বপূর্ণ।

যদি recent conversation-এ Shawon এবং Mitu-এর মধ্যে
কোনো বিষয় নিয়ে কথা হয়ে থাকে, current message-এর
meaning বোঝার সময় সেটাকে গুরুত্ব দেবে।

কিন্তু recent Mitu reply-কে template হিসেবে copy করবে না।

---------------------------------------------------------
STYLE LEARNING
---------------------------------------------------------

পুরোনো conversation থেকে নিচের বিষয়গুলো শিখবে:

- Mitu সাধারণত কত ছোট/বড় reply দিত
- Bangla নাকি Banglish বেশি ব্যবহার করত
- emoji ব্যবহার করত কিনা
- playful ছিল কিনা
- serious হলে কেমন বলত
- প্রশ্নের উত্তর কীভাবে দিত
- Shawon-এর message-এর প্রতি কী ধরনের reaction দিত

কিন্তু এগুলো থেকে নতুন reply তৈরি করবে।

---------------------------------------------------------
FINAL OUTPUT
---------------------------------------------------------

শুধু final conversational reply দেবে।

কোনো analysis নয়।
কোনো explanation নয়।
কোনো heading নয়।
কোনো numbered list নয়।
কোনো "thinking" নয়।

সরাসরি Mitu-style reply।
""".strip()


# =========================================================
# BUILD STYLE EXAMPLES
# =========================================================

def build_style_examples(examples):

    if not examples:
        return ""


    lines = []

    count = 0


    for item in examples:

        if not isinstance(item, dict):
            continue


        user_text = clean_text(
            item.get("user", "")
        )


        mitu_text = clean_text(
            item.get("mitu", "")
        )


        if not user_text or not mitu_text:
            continue


        if is_meta_text(mitu_text):
            continue


        lines.append(
            "Shawon: " + user_text
        )


        lines.append(
            "Mitu: " + mitu_text
        )


        lines.append("")


        count += 1


        if count >= 8:
            break


    if not lines:
        return ""


    return (
        "পুরোনো WhatsApp conversation-এর কিছু "
        "style examples নিচে দেওয়া হলো।\n\n"
        "এগুলো শুধু Mitu-এর ভাষা ও tone শেখার জন্য। "
        "কোনো answer copy করা যাবে না।\n\n"
        +
        "\n".join(lines)
    )


# =========================================================
# BUILD RECENT CONTEXT
# =========================================================

def build_recent_context(recent_chat):

    if not recent_chat:
        return ""


    lines = []


    for item in recent_chat[-12:]:

        if not isinstance(item, dict):
            continue


        role = item.get(
            "role",
            ""
        )


        content = clean_text(
            item.get(
                "content",
                ""
            )
        )


        if not content:
            continue


        if is_meta_text(content):
            continue


        if role == "user":

            lines.append(
                "Shawon: " + content
            )


        elif role == "assistant":

            lines.append(
                "Mitu: " + content
            )


    if not lines:
        return ""


    return (
        "এটাই current conversation-এর recent context।\n"
        "এটি current message বোঝার জন্য ব্যবহার করবে।\n\n"
        +
        "\n".join(lines)
    )


# =========================================================
# BUILD MESSAGES
# =========================================================

def build_messages(data: ChatRequest):

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
    # STYLE EXAMPLES
    # =====================================================

    style_context = build_style_examples(
        data.examples
    )


    if style_context:

        messages.append({

            "role": "system",

            "content":
                style_context
        })


    # =====================================================
    # RECENT CHAT
    # =====================================================

    recent_context = build_recent_context(
        data.recent_chat
    )


    if recent_context:

        messages.append({

            "role": "system",

            "content":
                recent_context
        })


    # =====================================================
    # CURRENT MESSAGE
    # =====================================================

    current_message = clean_text(
        data.message
    )


    messages.append({

        "role": "user",

        "content":
            current_message
    })


    return messages


# =========================================================
# CLEAN AI REPLY
# =========================================================

def clean_ai_reply(text):

    if not text:
        return ""


    text = clean_text(
        text
    )


    # -----------------------------------------------------
    # Remove code fences
    # -----------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        if len(lines) >= 3:

            text = "\n".join(
                lines[1:-1]
            ).strip()


    # -----------------------------------------------------
    # Remove meta prefixes
    # -----------------------------------------------------

    bad_prefixes = [

        "Here's a thinking process:",

        "Here is a thinking process:",

        "The user is asking:",

        "The user is asking",

        "According to the instructions:",

        "According to the instructions",

        "Analysis:",

        "Thinking:",

        "Reasoning:",

        "Identify context:",

        "Analyze user input:"
    ]


    changed = True


    while changed:

        changed = False


        for prefix in bad_prefixes:

            if text.lower().startswith(
                    prefix.lower()
            ):

                parts = text.split(
                    "\n",
                    1
                )


                if len(parts) == 2:

                    text = parts[1].strip()

                else:

                    text = ""


                changed = True

                break


    # -----------------------------------------------------
    # Remove accidental speaker prefix
    # -----------------------------------------------------

    prefixes = [

        "Mitu:",

        "Mitu -",

        "Mitu—"
    ]


    for prefix in prefixes:

        if text.lower().startswith(
                prefix.lower()
        ):

            text = text[
                len(prefix):
            ].strip()


            break


    return text.strip()


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post("/v1/mitu/chat")
async def mitu_chat(
        data: ChatRequest
):

    message = clean_text(
        data.message
    )


    if not message:

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


    # =====================================================
    # BUILD PROMPT
    # =====================================================

    messages = build_messages(
        data
    )


    # =====================================================
    # PAYLOAD
    # =====================================================

    payload = {

        "model":
            AI_MODEL,

        "messages":
            messages,

        "temperature":
            0.88,

        "top_p":
            0.92,

        "frequency_penalty":
            0.35,

        "presence_penalty":
            0.20,

        "max_tokens":
            180
    }


    # =====================================================
    # HEADERS
    # =====================================================

    headers = {

        "Authorization":
            "Bearer " + AI_API_KEY,

        "Content-Type":
            "application/json",

        "Accept":
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

        timeout = httpx.Timeout(

            connect=30.0,

            read=120.0,

            write=30.0,

            pool=30.0
        )


        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:

            response = await client.post(

                AI_BASE_URL +
                "/chat/completions",

                headers=headers,

                json=payload
            )


        # =================================================
        # PROVIDER ERROR
        # =================================================

        if (
                response.status_code < 200
                or
                response.status_code >= 300
        ):

            detail = response.text


            raise HTTPException(

                status_code=502,

                detail=
                    "AI provider error: "
                    +
                    detail
            )


        # =================================================
        # JSON
        # =================================================

        result = response.json()


        choices = result.get(
            "choices",
            []
        )


        if not choices:

            raise HTTPException(

                status_code=502,

                detail=
                    "AI returned no choices"
            )


        message_data = choices[0].get(
            "message",
            {}
        )


        reply = message_data.get(
            "content",
            ""
        )


        # Some providers may return
        # content as a structured value.

        if isinstance(
                reply,
                list
        ):

            parts = []


            for part in reply:

                if isinstance(
                        part,
                        dict
                ):

                    value = part.get(
                        "text",
                        ""
                    )


                    if value:

                        parts.append(
                            str(value)
                        )


            reply = "\n".join(
                parts
            )


        reply = clean_ai_reply(
            reply
        )


        if not reply:

            raise HTTPException(

                status_code=502,

                detail=
                    "AI returned empty response"
            )


        # =================================================
        # RESPONSE
        # =================================================

        return {

            "success":
                True,

            "reply":
                reply,

            "model":
                AI_MODEL
        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=502,

            detail=
                "AI provider connection error: "
                +
                str(e)
        )
