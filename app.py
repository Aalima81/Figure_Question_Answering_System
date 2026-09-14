import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient
import base64
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Figure Question Answering",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def get_client():

    client = InferenceClient(
        api_key=st.secrets["HF_TOKEN"]
    )

    return client


client = get_client()


# ============================================================
# IMAGE TO DATA URL
# ============================================================

def image_to_data_url(image):

    buffer = io.BytesIO()

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
# QUESTION ANSWERING
# ============================================================

def improved_answer(image, question):

    image = image.convert("RGB")

    image_url = image_to_data_url(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512
    )

    answer = response.choices[0].message.content

    return answer.strip()


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
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
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


        st.write(answer)
