import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient
import base64
from io import BytesIO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Figure Question Answering",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# HUGGING FACE CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# ============================================================
# GET HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def get_client():

    token = st.secrets["HF_TOKEN"]

    client = InferenceClient(
        provider="featherless-ai",
        api_key=token
    )

    return client


# ============================================================
# IMAGE TO BASE64
# ============================================================

def image_to_data_url(image):

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    image_bytes = buffer.getvalue()

    base64_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return f"data:image/png;base64,{base64_image}"


# ============================================================
# ANSWER FUNCTION
# ============================================================

def improved_answer(image, question):

    import base64
    from io import BytesIO

    # Convert PIL image to base64
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    # Strong instruction to force visual grounding
    prompt = f"""
You are a visual question-answering system.

Answer the user's question ONLY from the uploaded figure.

Carefully inspect:
- labels
- arrows
- boxes
- modules
- connections
- symbols
- text inside the figure
- the overall flow

Do NOT assume information that is not visible in the figure.
Do NOT give a generic cybersecurity explanation.
If the requested information is not visible, say:
"The figure does not provide enough information to answer this."

User question:
{question}

Give a clear and concise answer based specifically on the figure.
"""

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
                            "url": f"data:image/jpeg;base64,{image_base64}"
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
# UPLOAD IMAGE
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
