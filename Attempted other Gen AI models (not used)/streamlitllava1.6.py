import streamlit as st
import torch
import os
import tempfile
import fitz
import pdfplumber
import pandas as pd
from PIL import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from io import BytesIO
from torchvision import transforms

# ------------------------- PDF Extraction -------------------------

def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

def extract_tables_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    if len(df.columns) >= 2:
                        tables.append(df)
                except Exception:
                    continue
    return tables

def extract_images_pdf(pdf_path, save_folder="images"):
    os.makedirs(save_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        for j, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(BytesIO(base_image["image"])).convert("RGB")
            path = os.path.join(save_folder, f"page{i+1}_img{j+1}.png")
            image.save(path)
            image_paths.append(path)
    return image_paths

# ------------------------ Load LLaVA-Next ------------------------

@st.cache(allow_output_mutation=True)
def load_llava_next():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    return processor, model

# ✅ Manual image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    return transform(image).unsqueeze(0)  # [1, 3, 336, 336]

# ------------------------ Generate Lesson (One Image at a Time) ------------------------

def generate_lesson_one_by_one(text, image_paths):
    processor, model = load_llava_next()
    lesson_parts = []

    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")

        # ✅ Include <image> token
        # Prompt with <image>
        prompt = (
            "<|user|>\n"
            "<image>\n"
            f"{text}\n\n"
            f"You are an expert instructional designer. Based on the above text and the image shown (Image {i+1}), "
            f"write a relevant section of a structured educational lesson plan. Include section headings and reference this image clearly.\n"
            "<|assistant|>"
        )

        # Correctly pass a *list* of one image
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

        # Only keep inputs relevant to model.generate
        inputs = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask", "pixel_values"}}

        # Generate
        output = model.generate(**inputs, max_new_tokens=1024)
        generated = processor.batch_decode(output, skip_special_tokens=True)[0]
        lesson_parts.append(f"### 📷 Image {i+1}: {os.path.basename(image_path)}\n\n{generated}")

    return "\n\n---\n\n".join(lesson_parts)

# ------------------------ Streamlit UI ------------------------

st.set_page_config(page_title="📘 LLaVA-NeXT Lesson Generator", layout="wide")
st.title("📘 Gen AI Lesson Plan Generator with LLaVA-NeXT v1.6 Mistral")
st.write("Upload a PDF and get a structured AI-generated lesson plan using both text and images.")

uploaded_file = st.file_uploader("📎 Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("📄 Extracting PDF content..."):
        full_text = extract_text_pymupdf(tmp_path)
        tables = extract_tables_pdf(tmp_path)
        image_paths = extract_images_pdf(tmp_path, save_folder="images")

    with st.spinner("🧠 Generating lesson with LLaVA-NeXT..."):
        lesson = generate_lesson_one_by_one(full_text, image_paths)

    st.markdown("## 📝 AI-Generated Lesson Plan")
    st.markdown(lesson)

    if tables:
        st.markdown("## 📊 Extracted Tables")
        for i, table in enumerate(tables):
            st.markdown(f"**Table {i+1}**")
            st.dataframe(table)

    if image_paths:
        st.markdown("## 🖼️ Extracted Images")
        for i, path in enumerate(image_paths):
            st.image(path, caption=f"Image {i+1}", use_column_width=True)
