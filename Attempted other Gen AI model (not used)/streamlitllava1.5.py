import streamlit as st
import torch
import os
import tempfile
import fitz
import pdfplumber
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from io import BytesIO

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

# ------------------------ Load Quantized LLaVA ------------------------

@st.cache(allow_output_mutation=True)
def load_llava():
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    return processor, model

# ------------------------ Unified Lesson Generator ------------------------

def generate_unified_lesson(text, image_paths):
    processor, model = load_llava()

    # Load all images
    images = [Image.open(p).convert("RGB") for p in image_paths]
    image_prompts = "\n".join(["<image>" for _ in images])

    prompt = (
        "<|user|>\n"
        f"{image_prompts}\n"
        f"{text}\n\n"
        "You are a professional instructional designer. Generate a full educational lesson plan based on the above content and visuals.\n\n"
        "Structure your response with these parts:\n"
        "1. Title\n"
        "2. Learning Objectives\n"
        "3. Introduction\n"
        "4. Visual-Based Explanation (reference images where helpful)\n"
        "5. Key Takeaways\n"
        "6. Summary or Conclusion\n"
        "7. Assignment for Students\n\n"
        "Mention images by number using 'Image 1', 'Image 2', etc.\n"
        "<|assistant|>"
    )

    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=2048)
    generated = processor.batch_decode(output, skip_special_tokens=True)[0]

    return generated


# ------------------------ Streamlit UI ------------------------

st.set_page_config(page_title="📘 Unified Gen AI Lesson Generator", layout="wide")
st.title("📘 Unified Gen AI Lesson Plan Generator (with LLaVA)")
st.write("Upload a PDF. This app uses LLaVA to generate a **single lesson** from all the images and full text.")

uploaded_file = st.file_uploader("📎 Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("📄 Extracting text, tables, and images..."):
        full_text = extract_text_pymupdf(tmp_path)
        tables = extract_tables_pdf(tmp_path)
        image_paths = extract_images_pdf(tmp_path, save_folder="images")

    with st.spinner("🧠 Generating unified lesson using LLaVA..."):
        unified_lesson = generate_unified_lesson(full_text, image_paths)

    st.markdown("## 📝 Unified AI-Generated Lesson Plan")
    st.markdown(unified_lesson)

    if tables:
        st.markdown("## 📊 Extracted Tables")
        for i, table in enumerate(tables):
            st.markdown(f"**Table {i+1}**")
            st.dataframe(table)

    if image_paths:
        st.markdown("## 🖼️ Extracted Images")
        for i, path in enumerate(image_paths):
            st.image(path, caption=f"Image {i+1}", use_column_width=True)
