import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient
import base64
import io

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct:featherless-ai"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Figure Question Answering",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def load_client():

    if "HF_TOKEN" not in st.secrets:
        st.error("HF_TOKEN is missing from Streamlit Secrets.")
        st.stop()

    client = InferenceClient(
        provider="featherless-ai",
        api_key=st.secrets["HF_TOKEN"]
    )

    return client

client = load_client()


# ============================================================
# ANSWER FUNCTION
# ============================================================

def improved_answer(image, question):

    image = image.convert("RGB")

    # Convert PIL image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    image_url = f"data:image/png;base64,{image_base64}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-3B-Instruct:featherless-ai",
        messages=messages,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()

