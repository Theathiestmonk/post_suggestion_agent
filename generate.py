# generate.py

from rl_agent import select_action
from prompt_template import TOPIC_GENERATOR,PROMPT_GENERATOR, TRENDY_TOPIC_PROMPT, classify_trend_style
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sklearn.decomposition import PCA
from db import recent_topics




load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_URL = os.getenv("GROK_API_URL")
GROK_API_KEY = os.getenv("GROK_URL_KEY")


# ============================================================
# LLM CLIENTS (ABSTRACTION LAYER)
# ============================================================
# Note: Models are initialized only when API keys are available

def call_gpt_4o_mini(prompt: str) -> dict:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize the model with the API key
    gpt_4o_mini = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    response = gpt_4o_mini.invoke([
        HumanMessage(content=prompt)
    ])

    try:
        return json.loads(response.content)
    except Exception:
        raise ValueError("GPT-4o mini did not return valid JSON")


def call_grok(prompt: str) -> str | dict:
    """
    Grok HTTP client.
    - Reads env vars at RUNTIME (no import-time freeze)
    - Works with paid API keys
    - Returns JSON if possible, else raw text
    """

    GROK_API_KEY = os.getenv("GROK_API_KEY")
    GROK_API_URL = os.getenv("GROK_API_URL")

    if not GROK_API_KEY:
        raise ValueError("GROK_API_KEY not found in environment variables")

    if not GROK_API_URL:
        raise ValueError("GROK_API_URL not found in environment variables")

    try:
        response = requests.post(
            GROK_API_URL,
            headers={
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                # safest + cheapest + API-enabled
                "model": "grok-4-1-fast-non-reasoning",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            },
            timeout=30
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Grok request failed: {e}")

    if response.status_code != 200:
        raise RuntimeError(
            f"Grok API error {response.status_code}: {response.text}"
        )

    content = response.json()["choices"][0]["message"]["content"]

    # Prompt-generation → JSON
    # Topic-generation → plain text
    try:
        return json.loads(content)
    except Exception:
        return content



# ============================================================
# TOPIC GENERATOR
# ============================================================

def generate_topic(
    business_context: str,
    platform: str,
    date: str,
    business_id: str = None,

) -> dict:
    """
    Generates a post topic using Grok.
    Returns:
    {
      "topic": str,
      "reasoning": str
    }
    """
    print(f"Business context: {business_context}")
    filled_prompt = TOPIC_GENERATOR
    filled_prompt = filled_prompt.replace("{{BUSINESS_CONTEXT}}", business_context)
    filled_prompt = filled_prompt.replace("{{PLATFORM}}", platform)
    filled_prompt = filled_prompt.replace("{{DATE}}", date)
    filled_prompt = filled_prompt.replace("{{RECENT_TOPICS}}", str(recent_topics(business_id,platform)))

    try:
        response = call_grok(filled_prompt)
        return {
            "topic": response
        }

    except Exception as e:
        print(f"Error generating topic: {e}")
        return {
            "topic": "Get to know our brand and what we do, Give a brief introduction to the business and what we do "
        }

    

# ============================================================
# EMBED TOPIC
# ============================================================



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def embed_topic(text: str) -> np.ndarray:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    
    return np.array(response.data[0].embedding, dtype=np.float32)

# ============================================================
# CONTEXT BUILDER 
# ============================================================

def build_context(business_embedding, topic_embedding, platform, time):
    """
    Build RL context from embeddings and scheduling info.
    """
    return {
        "platform": platform,
        "time_bucket": time,
        "business_embedding": business_embedding,
        "topic_embedding": topic_embedding
    }


# ============================================================
# MAIN GENERATION FUNCTION
# ============================================================

def generate_prompts(
    inputs: dict,
    business_embedding,
    topic_embedding,
    platform: str,
    time: str,
    topic_text: str,profile_data: dict,
    business_context: str
) -> dict:
    """
    Single execution point between RL and LLMs.
    """

    print(f"🤖 RL Context: Platform={platform}, Time={time}")

    # 1️⃣ Build RL context (using your build_context)
    context = build_context(
        business_embedding=business_embedding,
        topic_embedding=topic_embedding,
        platform=platform,
        time=time
    )

    # 2️⃣ RL decides creative controls
    action, ctx_vec = select_action(context) 
    
    print(f"🎯 RL Selected Action: {action}")
    hook_type = action.get("HOOK_TYPE", "")

    # 3️⃣ Merge inputs + RL action for placeholders
    merged = {**inputs, **action}

    # =====================================================
    # 🔥 TRENDY → GROK
    # =====================================================
    if hook_type == "trendy topic hook":
        selected_style = classify_trend_style(
            inputs["BUSINESS_TYPES"],
            inputs["INDUSTRIES"]
        )

        filled_prompt = TRENDY_TOPIC_PROMPT
        merged_with_style = {
            **merged,
            "selected_style": selected_style,
            "BUSINESS_CONTEXT": business_context,
            "BUSINESS_AESTHETIC": profile_data["brand_voice"],
            "BUSINESS_TYPES": profile_data["business_types"],
            "INDUSTRIES": profile_data["industries"],
            "topic_text": topic_text
        }

        for k, v in merged_with_style.items():
            if isinstance(v, list):
                v = ", ".join(v)
            filled_prompt = filled_prompt.replace(f"{{{{{k}}}}}", str(v))

        print(f"📝 Sending to Grok (Trendy Topic): {filled_prompt[:200]}...")
        llm_response = call_grok(filled_prompt)

        if not isinstance(llm_response, dict):
            print("⚠️ Grok returned non-JSON output. Raw response:")
            print(llm_response)
            raise ValueError("Invalid Grok response format")

        if "caption_prompt" not in llm_response or "image_prompt" not in llm_response:
            raise ValueError(f"Incomplete Grok response: {llm_response}")

        print(f"📝 Generated Caption Prompt: {llm_response['caption_prompt']}\n")
        print(f"📝 Generated Image Prompt: {llm_response['image_prompt']}\n")

        return {
            "mode": "trendy",
            "caption_prompt": llm_response["caption_prompt"],
            "image_prompt": llm_response["image_prompt"],
            "style": selected_style,
            "action": action,
            "context": context,
            "ctx_vec": ctx_vec
        }


    # =====================================================
    # ✅ NON-TRENDY → GPT-4o MINI
    # =====================================================
    filled_prompt = PROMPT_GENERATOR
    merged = {
    **inputs,
    **action,
    "BUSINESS_CONTEXT": business_context,
    "BUSINESS_AESTHETIC": profile_data["brand_voice"],
    "BUSINESS_TYPES": profile_data["business_types"],
    "INDUSTRIES": profile_data["industries"],
    "topic_text": topic_text
}

    for k, v in merged.items():
        if isinstance(v, list):
            v = ", ".join(v)
        filled_prompt = filled_prompt.replace(f"{{{{{k}}}}}", str(v))

    print(f"📝 Sending to GPT-4o-mini (Standard)")
    llm_response = call_gpt_4o_mini(filled_prompt)
    print(f"📝 Generated Caption Prompt: {llm_response['caption_prompt'][:180]}...")
    print(f"📝 Generated Image Prompt: {llm_response['image_prompt'][:180]}...")

    return {
        "mode": "standard",
        "caption_prompt": llm_response["caption_prompt"],
        "image_prompt": llm_response["image_prompt"],
        "action": action,
        "context": context,
        "ctx_vec": ctx_vec
    }