
import base64
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
from openai import OpenAI

# Ensure you have set OPENAI_API_KEY as an environment variable or use client = OpenAI(api_key="redacted")
client = OpenAI(api_key="redacted")

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


def query_gpt4v(image_path, prompt="Describe this image."):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content


def generate_lesson_from_extracted_data(pdf_path):
    prompt = "Generate a lesson plan based on the content in this image."

    # Extract first image
    extracted_images = extract_images_pdf(pdf_path)
    if not extracted_images:
        return "No image found", [], []

    image_path = extracted_images[0]  # Use the first image for GPT-4V
    lesson = query_gpt4v(image_path, prompt)

    # Extract tables and return markdown versions
    tables = extract_tables_pdf(pdf_path)
    tables_md = [table.to_markdown(index=False) for table in tables]

    return lesson, tables_md, extracted_images


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



# ------------------- GPT-4V Interface -------------------
st.header("GPT-4V Image + Prompt Interface")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
user_prompt = st.text_input("Enter a prompt for GPT-4V", value="Describe this image.")

if uploaded_image is not None and user_prompt:
    # Save image temporarily
    temp_image_path = os.path.join(tempfile.gettempdir(), uploaded_image.name)
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_image.read())

    # Query GPT-4V
    st.info("Querying GPT-4V...")
    try:
        gpt4v_response = query_gpt4v(temp_image_path, prompt=user_prompt)
        st.subheader("GPT-4V Response")
        st.write(gpt4v_response)
    except Exception as e:
        st.error(f"Error querying GPT-4V: {e}")
