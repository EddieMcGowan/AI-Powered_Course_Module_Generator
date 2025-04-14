
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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

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

# ------------------- Prompting Logic -------------------

def table_to_markdown(tables):
    return [df.to_markdown(index=False) for df in tables]

def get_context_block(text, tables_markdown, images):
    joined_tables_md = "\n\n".join(tables_markdown)
    joined_images = ", ".join(images)
    return (
        f"### Text:\n{text[:3000]}\n\n"
        f"### Tables (Markdown format):\n{joined_tables_md}\n\n"
        f"### Images (filenames):\n{joined_images}\n\n"
        "### Instructions:\n"
        "- Write an engaging lesson plan with sections and subheadings.\n"
        "- Embed tables by referencing [Table 1], [Table 2], etc.\n"
        "- Embed images using ![image1](...), ![image2](...), etc.\n"
        "- Place unused images/tables under '📎 Visuals & Extras'.\n"
    )

def generate_lesson_from_prompt(user_prompt, context_block):
    full_prompt = f"{user_prompt.strip()}\n\n{context_block}"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        torch_dtype=torch.float16
    )

    input_ids = tokenizer(full_prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)
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
    return lesson

# ------------------- Display & Download -------------------

def render_markdown_with_visuals(lesson: str, tables_md: list, images: list):
    used_tables = set()
    used_images = set()
    for i, table_md in enumerate(tables_md):
        tag = f"[Table {i+1}]"
        if tag in lesson:
            lesson = lesson.replace(tag, f"\n\n{table_md}\n\n")
            used_tables.add(i)

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

    unused_tables = [tables_md[i] for i in range(len(tables_md)) if i not in used_tables]
    unused_images = [images[i] for i in range(len(images)) if i not in used_images]

    if unused_tables or unused_images:
        st.markdown("---\n### 📎 Visuals & Extras")
        for i, table in enumerate(unused_tables):
            st.markdown(f"**[Unused Table {i+1}]**\n\n{table}")
        for i, path in enumerate(unused_images):
            st.image(path, caption=f"Unused image {i+1}", use_column_width=True)

    return unused_tables, unused_images

def download_as_pdf(lesson, unused_tables, unused_images, filename="lesson_plan.pdf"):
    output_path = os.path.join(tempfile.gettempdir(), filename)
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    x_margin = 1 * inch
    y = height - 1 * inch
    line_height = 14
    max_lines = int((height - 2 * inch) / line_height)

    def write_line(text):
        nonlocal y
        if y <= 1 * inch:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 1 * inch
        c.drawString(x_margin, y, text)
        y -= line_height

    c.setFont("Helvetica", 11)

    # Lesson content
    for line in lesson.split("\n"):
        write_line(line)

    write_line("")
    write_line("📎 Visuals & Extras")
    write_line("")

    for i, table in enumerate(unused_tables):
        write_line(f"[Unused Table {i+1}]")
        for tline in table.split("\n"):
            write_line(tline)
        write_line("")

    for i, img_path in enumerate(unused_images):
        try:
            img = Image.open(img_path)
            img.thumbnail((400, 400))
            if y <= 2.5 * inch:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 1 * inch
            c.drawImage(ImageReader(img), x_margin, y - 200, width=3.5*inch, preserveAspectRatio=True, mask='auto')
            write_line(f"Unused Image {i+1}: {os.path.basename(img_path)}")
            y -= 210
        except Exception as e:
            write_line(f"[Could not load image {img_path}]")

    c.save()
    return output_path

# ------------------- Streamlit App -------------------

st.set_page_config(page_title="Gen AI Lesson Generator", layout="wide")
st.title("📘 Gen AI-Powered Lesson Plan Generator")

DEFAULT_USER_PROMPT = "You are an educational content generator. Create a structured lesson based on the text, tables, and images attached."

if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = DEFAULT_USER_PROMPT
if "context_block" not in st.session_state:
    st.session_state.context_block = ""
if "tables_md" not in st.session_state:
    st.session_state.tables_md = []
if "images" not in st.session_state:
    st.session_state.images = []
if "last_lesson" not in st.session_state:
    st.session_state.last_lesson = ""
if "unused_tables" not in st.session_state:
    st.session_state.unused_tables = []
if "unused_images" not in st.session_state:
    st.session_state.unused_images = []

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    combined_text = ""
    all_tables = []
    all_images = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            path = tmp_file.name
            combined_text += extract_text_pymupdf(path) + "\n"
            all_tables.extend(extract_tables_pdf(path))
            all_images.extend(extract_images_pdf(path))

    st.session_state.tables_md = table_to_markdown(all_tables)
    st.session_state.images = all_images
    st.session_state.context_block = get_context_block(combined_text, st.session_state.tables_md, st.session_state.images)

    st.session_state.user_prompt = st.text_area(
        "✏️ Customize your lesson prompt:",
        value=st.session_state.user_prompt,
        height=200
    )

    if st.button("Generate Lesson"):
        with st.spinner("Generating..."):
            lesson = generate_lesson_from_prompt(st.session_state.user_prompt, st.session_state.context_block)
        st.session_state.last_lesson = lesson
        st.markdown("## 🧠 Lesson Plan")
        unused_tables, unused_images = render_markdown_with_visuals(lesson, st.session_state.tables_md, st.session_state.images)
        st.session_state.unused_tables = unused_tables
        st.session_state.unused_images = unused_images

    if st.session_state.last_lesson:
        st.markdown("### 🔁 Reprompt Lesson")
        reprompt_input = st.text_area(
            "Modify and reprompt:",
            value=st.session_state.user_prompt,
            height=200,
            key="reprompt_input"
        )
        if st.button("Reprompt with changes"):
            with st.spinner("Reprompting..."):
                lesson = generate_lesson_from_prompt(reprompt_input, st.session_state.context_block)
            st.session_state.user_prompt = reprompt_input
            st.session_state.last_lesson = lesson
            st.markdown("## 🧠 Updated Lesson Plan")
            unused_tables, unused_images = render_markdown_with_visuals(lesson, st.session_state.tables_md, st.session_state.images)
            st.session_state.unused_tables = unused_tables
            st.session_state.unused_images = unused_images

        pdf_path = download_as_pdf(
            st.session_state.last_lesson,
            st.session_state.unused_tables,
            st.session_state.unused_images
        )
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Lesson (PDF with Images)", f, file_name="lesson_plan.pdf")
