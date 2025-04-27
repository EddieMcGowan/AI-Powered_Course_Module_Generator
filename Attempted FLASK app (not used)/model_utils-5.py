import os
import pdfplumber
import pandas as pd
import fitz
def extract_text_pymupdf(pdf_path):
    """Extracts text from a structured (non-scanned) PDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_tables_pdf(pdf_path):
    """Extracts tables from a PDF and converts them into DataFrames."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                df = pd.DataFrame(table[1:], columns=table[0])  # First row as headers
                tables.append(df)
    return tables

import os
import fitz  # PyMuPDF
from PIL import Image
import io

def extract_images_pdf(pdf_path, save_folder="images"):
    """Extracts embedded images (not full pages) from a PDF and saves them as PNG files."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            filename = f"{save_folder}/page{page_index+1}_img{img_index+1}.{image_ext}"
            image.save(filename)
            image_paths.append(filename)

    return image_paths

import requests
from bs4 import BeautifulSoup

def extract_text_webpage(url):
    """Extracts text from an article webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = "\n".join([p.get_text(strip=True) for p in soup.find_all("p")])
    return text

def extract_tables_webpage(url):
    """Extracts tables from a webpage and converts them into DataFrames."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            rows.append(cells)
        if rows:
            df = pd.DataFrame(rows[1:], columns=rows[0])  # First row as headers
            tables.append(df)
    return tables

def extract_images_webpage(url, save_folder="images"):
    """Extracts images from a webpage and saves them locally."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    images = []
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
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
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")
    return images

def extract_text(source):
    """Extracts text from PDF or web URL."""
    if source.endswith(".pdf"):
        return extract_text_pymupdf(source)  # Extract from PDF
    elif source.startswith("http"):
        return extract_text_webpage(source)  # Extract from web page
    else:
        raise ValueError("Unsupported file type. Provide a PDF or URL.")

def extract_tables(source):
    """Extracts tables from a PDF or web URL and converts them into DataFrames."""
    if source.endswith(".pdf"):
        return extract_tables_pdf(source)  # Extract from PDF
    elif source.startswith("http"):
        return extract_tables_webpage(source)  # Extract from web page
    else:
        raise ValueError("Unsupported file type. Provide a PDF or URL.")

def extract_images(source, save_folder="images"):
    """Extracts images from PDFs or web pages and saves them locally."""
    if source.endswith(".pdf"):
        return extract_images_pdf(source, save_folder)  # Extract from PDF
    elif source.startswith("http"):
        return extract_images_webpage(source, save_folder)  # Extract from web page
    else:
        raise ValueError("Unsupported file type. Provide a PDF or URL.")

from transformers import pipeline

def extract_all(sources):
    """Extracts text, tables, and images from multiple PDFs and web pages."""
    combined_text = []
    combined_tables = []
    combined_images = []
    source_references = []  # Store source details

    for source in sources:
        text = extract_text(source)
        tables = extract_tables(source)
        images = extract_images(source)

        combined_text.append(text)
        combined_tables.extend(tables)
        combined_images.extend(images)
        source_references.append(source)

    return combined_text, combined_tables, combined_images, source_references

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

def generate_lesson_from_extracted_data(sources):
    """
    Generates a lesson from extracted text, tables, and image filenames.
    """

    # Step 1: Extract all content
    all_texts, all_tables, all_images, all_sources = extract_all(sources)

    # Step 2: Convert tables to Markdown
    def table_to_markdown(tables):
        markdown = []
        for df in tables:
            try:
                markdown.append(df.to_markdown(index=False))
            except Exception as e:
                markdown.append(f"*Table rendering failed: {e}*")
        return "\n\n".join(markdown)

    tables_markdown = table_to_markdown(all_tables)

    # Step 3: Combine and truncate extracted text
    combined_text = "\n\n".join(all_texts)[:3000]  # Safe chunk for token limit

    # Step 4: Prompt assembly
    prompt = f"""
You are an educational content generator. Create a structured and engaging lesson plan using the provided text, tables, and images.

### Text:
{combined_text}

### Tables (Markdown format):
{tables_markdown}

### Images (filenames only):
{', '.join(all_images)}

### Instructions:
- Use the **text** as the main narrative for the lesson. Organize it into logical **sections** and **subheadings**.
- Integrate each **table** at a relevant point using HTML or Markdown, and briefly explain what it shows.
- For **each image filename**, reference it **in context** using the Markdown format:  
  `![image1](images/pdf_page_11_img1.png)` — and provide a brief caption or explanation of what the image shows.
- If you're unsure where an image fits, include it at the end under a section called ** Visuals**.

Generate the full lesson content below:

"""

    # Step 6: Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Step 7: Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)

    # Step 8: Generate output
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            return_dict_in_generate=True
        )

    generated_ids = output.sequences[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text, all_texts, all_tables, all_images, all_sources

import re
from IPython.display import Markdown, display
def render_output_with_images_and_tables(lesson, images, table_blocks):
    import pandas as pd
    from io import StringIO
    import os
    import re

    # Replace markdown-style image links with actual HTML <img> tags
    lesson = re.sub(
    r'!\[.*?\]\((.*?)\)',
    lambda m: f'<img src="/{m.group(1)}" width="400">',
    lesson)

    # Convert markdown-like tables to HTML
    html_tables = []
    for i, table_md in enumerate(table_blocks):
        try:
            if "<table" in table_md.lower():  # Already HTML
                html_tables.append(f"<h4>Table {i+1}</h4>" + table_md)
                continue

            # Otherwise treat as markdown
            cleaned_md = "\n".join(
                line for line in table_md.splitlines()
                if not re.match(r"^\s*\|?\s*[:\-]+[\s:\-|\+]*$", line.strip())
            )

            cleaned_md = cleaned_md.replace("|", ",")  # Turn into CSV format
            df = pd.read_csv(StringIO(cleaned_md), engine="python")
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = [col.strip().strip(":") for col in df.columns]

            html_table = df.to_html(classes="styled-table", index=False, border=1)
            html_tables.append(f"<h4>Table {i+1}</h4>" + html_table)
        except Exception as e:
            html_tables.append(f"<p><strong>Error rendering Table {i+1}:</strong> {e}</p>")

    # Append tables and images to the end
    combined = lesson
    for html_table in html_tables:
        combined += "\n\n" + html_table

    return combined


def display_images(images):
    for img_path in images:
        if os.path.exists(img_path):
            print(f"📸 {os.path.basename(img_path)}")
            display(IPImage(filename=img_path))
        else:
            print(f"⚠️ Could not find image: {img_path}")

import ipywidgets as widgets
from IPython.display import display, clear_output
# --- Load model if not already loaded ---
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

# --- Reprompt function ---
def reprompt_lesson(original_lesson: str, instruction: str, temperature: float = 0.7) -> str:
    prompt = f"{instruction.strip()}\n\nOriginal Lesson Plan:\n{original_lesson.strip()}"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            return_dict_in_generate=True
        )

    generated_ids = output.sequences[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# --- Widgets ---
instruction_box = widgets.Text(
    value='',
    placeholder='Enter instruction or type "approved"',
    description='Instruction:',
    layout=widgets.Layout(width='100%')
)

submit_button = widgets.Button(description="Submit", button_style='success')
output_box = widgets.Output()

# --- Logic to handle button click ---
def on_submit_clicked(b):
    global current_lesson
    with output_box:
        clear_output(wait=True)
        instruction = instruction_box.value.strip()
        if instruction.lower() == "approved":
            print("✅ Lesson approved by the teacher.")
        else:
            # Generate and display revised lesson
            revised_lesson = reprompt_lesson(current_lesson, instruction)
            current_lesson = revised_lesson  # update for next iteration
            revised_final = render_output_with_images_and_tables(revised_lesson, images, table_blocks)
            print("📝 Revised Lesson:\n")
            print(revised_final)

submit_button.on_click(on_submit_clicked)

# --- Display the input and button ---
display(widgets.VBox([instruction_box, submit_button, output_box]))

# In[ ]:

from flask import Flask, render_template, request, redirect, send_file

import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    lesson = ""
    if request.method == "POST":
        file = request.files["pdf"]
        instruction = request.form.get("instruction", "")
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Process and generate lesson
        text, tables, images = process_pdf(filepath)
        if "lesson" not in request.form:
            lesson = generate_lesson(text, tables, images)
        else:
            lesson = reprompt_lesson(request.form["lesson"], instruction)

        return render_template("index.html", lesson=lesson, filename=file.filename)

    return render_template("index.html")

    app.run(debug=True)

def process_doc(doc_path):
    """
    Extracts content from the PDF and generates the first version of the lesson.

    Returns:
        lesson (str): Raw generated lesson text
        lesson_final (str): Rendered lesson with tables/images
        table_blocks (list of str): Markdown tables
        images (list of str): Image paths
    """
    #text, tables, images, _ = extract_all([pdf_path])
    lesson, text, tables, images, _ = generate_lesson_from_extracted_data([doc_path])
    table_blocks = [df.to_markdown(index=False) for df in tables]
    lesson_final = render_output_with_images_and_tables(lesson, images, table_blocks)
    return  lesson, lesson_final, table_blocks, images
