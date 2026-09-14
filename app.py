import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient
import base64
from io import BytesIO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=st.secrets["HF_TOKEN"],
    provider="auto"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Figure Question Answering",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# ANSWER FUNCTION
# ============================================================

def improved_answer(image, question):

    # Convert image to JPEG
    buffer = BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG"
    )

    # Convert image to Base64
    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    # Prompt
    prompt = f"""
You are a visual question-answering system.

Look carefully at the uploaded figure and answer the
question ONLY using information visible in the figure.

Pay attention to:
- labels
- arrows
- boxes
- modules
- diagrams
- symbols
- text
- relationships
- data flow

Do not give a generic explanation from your own knowledge.

Do not assume that the figure contains information
that is not actually visible.

Question:
{question}

Give a clear, accurate and concise answer based specifically
on the uploaded figure.
"""

    # Hugging Face multimodal request
    response = client.chat_completion(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + image_base64
                            )
                        }
                    }
                ]
            }
        ],
        max_tokens=512,
        temperature=0.1
    )

    return response.choices[0].message.content


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🖼️ Figure Question Answering")

st.write(
    "Upload a figure and ask a question about the figure."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Figure",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# QUESTION
# ============================================================

question = st.text_area(
    "Enter your question",
    placeholder="Example: What does this figure show?"
)


# ============================================================
# DISPLAY IMAGE
# ============================================================

image = None

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Figure")

    st.image(
        image,
        use_container_width=True
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

if st.button(
    "Generate Answer",
    type="primary"
):

    if uploaded_file is None:

        st.warning(
            "Please upload a figure first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Qwen2.5-VL is analyzing the figure..."
        ):

            try:

                answer = improved_answer(
                    image,
                    question
                )

                st.subheader("Answer")

                st.write(answer)

            except Exception as e:

                st.error(
                    "An error occurred while generating the answer."
                )

                st.exception(e)
