import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


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
        api_key=st.secrets["HF_TOKEN"]
    )

    return client


client = load_client()


# ============================================================
# ANSWER FUNCTION
# ============================================================

def improved_answer(image, question):

    image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    response = client.chat_completion(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()


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
