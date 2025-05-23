import streamlit as st
import fitz
import pdfplumber
import pandas as pd
from PIL import Image
import io
import os
import tempfile
import torch
import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import textwrap  
# Using the same functions as the jupyter notebook
# I wrote documents in my github explainig these functions in the documents folder
# Extraction Utilities

#extract text from pdf
def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

#extract tables from pdf
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

# extract images from pdf
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

# extract text from webpages
def extract_text_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = "\n".join([p.get_text(strip=True) for p in soup.find_all("p")])
    return text

# extract tables form webpages
def extract_tables_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            rows.append(cells)
        if rows:
            header = rows[0]
            body = rows[1:]
            # Only keep rows where number of cells matches header
            clean_body = [r for r in body if len(r) == len(header)]
            if clean_body:
                try:
                    df = pd.DataFrame(clean_body, columns=header)
                    tables.append(df)
                except Exception as e:
                    print(f"Skipping a table due to error: {e}")
    return tables

# extract images from weppages
def extract_images_webpage(url, save_folder="images"):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    images = []
    os.makedirs(save_folder, exist_ok=True)
    for i, img in enumerate(soup.find_all("img")):
        img_url = img.get("src")
        if img_url and img_url.startswith(("http", "//")):
            img_url = img_url if img_url.startswith("http") else "https:" + img_url
            try:
                img_data = requests.get(img_url).content
                img_filename = os.path.join(save_folder, f"image_{i+1}.jpg")
                with open(img_filename, "wb") as f:
                    f.write(img_data)
                images.append(img_filename)
            except Exception:
                continue
    return images

# combine the text extraction from pdf and webpages
def extract_text(source):
    if source.endswith(".pdf"):
        return extract_text_pymupdf(source)
    elif source.startswith("http"):
        return extract_text_webpage(source)
    else:
        raise ValueError("Unsupported file type. Provide a PDF or URL.")

# combine the table extraction from pdf and webpages
def extract_tables(source):
    if source.endswith(".pdf"):
        return extract_tables_pdf(source)
    elif source.startswith("http"):
        return extract_tables_webpage(source)
    else:
        raise ValueError("Unsupported file type. Provide a PDF or URL.")

# combine the image extraction from pdf and webpages
def extract_images(source, save_folder="images"):
    if source.endswith(".pdf"):
        return extract_images_pdf(source, save_folder)
    elif source.startswith("http"):
        return extract_images_webpage(source, save_folder)
    else:
        raise ValueError("Unsupported file type. Provide a PDF or URL.")

#  Prompting Logic 

# Convert tables to markdown
def table_to_markdown(tables):
    return [df.to_markdown(index=False) for df in tables]

# Context to provide to the model
def get_context_block(text, tables_markdown, images):
    joined_tables_md = "\n\n".join(tables_markdown)
    joined_images = ", ".join(images)
    return (
        f"### Text:\n{text[:2000]}\n\n"
        f"### Tables (Markdown format):\n{joined_tables_md}\n\n"
        f"### Images (filenames):\n{joined_images}\n\n"
        "### Instructions:\n"
        "- Write an engaging lesson plan with sections and subheadings.\n"
        "- Embed tables by referencing [Table 1], [Table 2], etc.\n"
        "- Embed images using ![image1](...), ![image2](...), etc.\n"
        "- Place unused images/tables under '📎 Visuals & Extras'.\n"
    )

# Generate the lesson
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


# Render the tables and images in the lesson output
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
    # place the unsued tables and images at the bottom of the file
    unused_tables = [tables_md[i] for i in range(len(tables_md)) if i not in used_tables]
    unused_images = [images[i] for i in range(len(images)) if i not in used_images]

    if unused_tables or unused_images:
        st.markdown("---\n### 📎 Visuals & Extras")
        for i, table in enumerate(unused_tables):
            st.markdown(f"**[Unused Table {i+1}]**\n\n{table}")
        for i, path in enumerate(unused_images):
            st.image(path, caption=f"Unused image {i+1}", use_column_width=True)

    return unused_tables, unused_images

# download as a pdf
def download_as_pdf(lesson, unused_tables, unused_images, filename="lesson_plan.pdf"):
    output_path = os.path.join(tempfile.gettempdir(), filename)
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    x_margin = 1 * inch
    y = height - 1 * inch
    line_height = 14

    # dynamically wrap text in the pdf
    def write_line(text):
        nonlocal y
        # Dynamically calculate max characters that fit per line
        max_chars = int((width - 2 * x_margin) / (6 * 0.75))  
        wrapped_lines = textwrap.wrap(text, width=max_chars)
        for line in wrapped_lines:
            if y <= 1 * inch:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 1 * inch
            c.drawString(x_margin, y, line)
            y -= line_height

    c.setFont("Helvetica", 11)

    # Write lesson content
    for line in lesson.split("\n"):
        write_line(line)

    write_line("")  # Add a blank line
    write_line("📎 Visuals & Extras")
    write_line("")

    # Display unused tables
    for i, table in enumerate(unused_tables):
        write_line(f"[Unused Table {i+1}]")
        for tline in table.split("\n"):
            write_line(tline)
        write_line("")

    # Display unused images
    for i, img_path in enumerate(unused_images):
        try:
            img = Image.open(img_path)
            img_width, img_height = img.size

            # Resize image to fit width if necessary
            max_width = 4 * inch  # you can adjust
            if img_width > max_width:
                scale = max_width / img_width
                img_width *= scale
                img_height *= scale

            # If not enough space for image, go to next page
            if y - img_height < 1 * inch:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 1 * inch

            # Draw image
            c.drawImage(ImageReader(img), x_margin, y - img_height, width=img_width, height=img_height, preserveAspectRatio=True, mask='auto')
            y = y - img_height - 20  # Move y down after image (+ 20 points spacing)

            # Write caption
            write_line(f"Unused Image {i+1}: {os.path.basename(img_path)}")
        except Exception:
            write_line(f"[Could not load image {img_path}]")

    c.save()
    return output_path


#  Streamlit App 

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
# Upload files
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
url_input = st.text_input("Or enter a webpage URL to extract content:")

# Combine the context from the files
if uploaded_files or url_input:
    combined_text = ""
    all_tables = []
    all_images = []
    sources = []

    if uploaded_files:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                path = tmp_file.name
                sources.append(path)

    if url_input:
        sources.append(url_input)

    for src in sources:
        combined_text += extract_text(src) + "\n"
        all_tables.extend(extract_tables(src))
        all_images.extend(extract_images(src))

    st.session_state.tables_md = table_to_markdown(all_tables)
    st.session_state.images = all_images
    st.session_state.context_block = get_context_block(combined_text, st.session_state.tables_md, st.session_state.images)

    st.session_state.user_prompt = st.text_area(
        "✏️ Customize your lesson prompt:",
        value=st.session_state.user_prompt,
        height=200
    )
    # Generate a lesson from the context provided
    if st.button("Generate Lesson"):
        with st.spinner("Generating..."):
            lesson = generate_lesson_from_prompt(st.session_state.user_prompt, st.session_state.context_block)
        st.session_state.last_lesson = lesson
        st.markdown("## 🧠 Lesson Plan")
        unused_tables, unused_images = render_markdown_with_visuals(lesson, st.session_state.tables_md, st.session_state.images)
        st.session_state.unused_tables = unused_tables
        st.session_state.unused_images = unused_images
    # Reprompt the lesson if requested by the user
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
        # Download as a pdf
        pdf_path = download_as_pdf(
            st.session_state.last_lesson,
            st.session_state.unused_tables,
            st.session_state.unused_images
        )
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Lesson (PDF with Images)", f, file_name="lesson_plan.pdf")
