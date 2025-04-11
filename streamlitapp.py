import streamlit as st
import fitz
import pdfplumber
import pandas as pd
from PIL import Image
import io
import os
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# ------------------- Extraction Utilities -------------------

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
            image = Image.open(io.BytesIO(base_image["image"]))
            path = os.path.join(save_folder, f"page{i+1}_img{j+1}.{base_image['ext']}")
            image.save(path)
            image_paths.append(path)
    return image_paths

# ------------------- Gen AI Lesson Generator -------------------

def table_to_markdown(tables):
    return [df.to_markdown(index=False) for df in tables]

def generate_lesson_from_extracted_data(pdf_path):
    # Extract from PDF
    text = extract_text_pymupdf(pdf_path)
    tables = extract_tables_pdf(pdf_path)
    images = extract_images_pdf(pdf_path)

    # Prep supporting markdown
    tables_markdown = table_to_markdown(tables)
    joined_tables_md = "\n\n".join(tables_markdown)
    joined_images = ", ".join(images)

    # Prompt
    prompt = (
        "You are an educational content generator. Create a structured lesson based on the text, tables, and images below.\n\n"
        f"### Text:\n{text[:3000]}\n\n"
        f"### Tables (Markdown format):\n{joined_tables_md}\n\n"
        f"### Images (filenames):\n{joined_images}\n\n"
        "### Instructions:\n"
        "- Write an engaging lesson plan with sections and subheadings.\n"
        "- Embed tables by referencing [Table 1], [Table 2], etc.\n"
        "- Embed images using ![image1](...), ![image2](...), etc.\n"
        "- Place unused images/tables under '📎 Visuals & Extras'.\n"
    )

    # Load LLaMA 2 model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        torch_dtype=torch.float16
    )

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

    generated_ids = output[0][input_ids.shape[-1]:]
    lesson = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return lesson, tables_markdown, images

# ------------------- Rendering with Visuals -------------------

def render_markdown_with_visuals(lesson: str, tables_md: list, images: list):
    used_tables = set()
    used_images = set()

    # Replace table tags
    for i, table_md in enumerate(tables_md):
        tag = f"[Table {i+1}]"
        if tag in lesson:
            lesson = lesson.replace(tag, f"\n\n{table_md}\n\n")
            used_tables.add(i)

    # Split and render line by line
    lines = lesson.split("\n")
    for line in lines:
        image_match = re.match(r"!\[image(\d+)\]\([^)]+\)", line.strip())
        if image_match:
            idx = int(image_match.group(1)) - 1
            if 0 <= idx < len(images):
                st.image(images[idx], use_column_width=True)
                used_images.add(idx)
        else:
            st.markdown(line)

    # Unused items
    unused = []
    for i, table in enumerate(tables_md):
        if i not in used_tables:
            unused.append(f"**[Unused Table {i+1}]**\n\n{table}")
    for i, path in enumerate(images):
        if i not in used_images:
            st.image(path, caption=f"Unused image{i+1}", use_column_width=True)

    if unused:
        st.markdown("---\n### Visuals & Extras")
        for u in unused:
            if isinstance(u, str):
                st.markdown(u)

# ------------------- Streamlit App -------------------

st.set_page_config(page_title="Gen AI Lesson Generator", layout="wide")
st.title("📘 Gen AI-Powered Lesson Plan Generator")
st.write("Upload a structured PDF and get a full AI-generated lesson plan with embedded content.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Generating your lesson..."):
        lesson, tables_md, images = generate_lesson_from_extracted_data(tmp_path)

    st.markdown("## Generated Lesson")
    render_markdown_with_visuals(lesson, tables_md, images)
