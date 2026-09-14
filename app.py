import streamlit as st
import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)


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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    st.info("Loading Qwen2.5-VL-3B model...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    return processor, model


processor, model = load_model()


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

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
        if hasattr(v, "to")
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512
        )

    generated_ids = [
        output_ids[i][len(inputs["input_ids"][i]):]
        for i in range(len(output_ids))
    ]

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return answer.strip()


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🖼️ Figure Question Answering")

st.write(
    "Upload a figure and ask a question about the figure."
)


# ------------------------------------------------------------
# Upload image
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Figure",
    type=["png", "jpg", "jpeg"]
)


# ------------------------------------------------------------
# Question
# ------------------------------------------------------------

question = st.text_area(
    "Enter your question",
    placeholder="Example: What does this figure show?"
)


# ------------------------------------------------------------
# Display uploaded image
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Generate answer
# ------------------------------------------------------------

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

            answer = improved_answer(
                image,
                question
            )

        st.subheader("Answer")

        st.write(answer)