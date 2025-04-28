# AI-Powered Course Module Generator

## Project Description
This project develops an AI chatbot that generates structured **course modules** based on teacher-provided resources (PDFs and webpage) and user inputs.  
The AI automates lesson creation, refines content based on teacher feedback, and dynamically updates a streamlit website. 
Supporting documents are included in the repository.

## Data Sources
- **Teacher-provided materials**: PDFs, articles, and web links.
- **Public websites**: For web scraping text, tables, and images.
- No external public dataset is required — the tool processes user-uploaded resources. I uploaded a scientific paper pdf and a webpage I used for testing.

## Packages Required
| Package | Purpose |
|:---|:---|
| `torch` | Load and run LLM (LLaVA or Vicuna) |
| `transformers` | Tokenization and generation with LLM |
| `PyMuPDF (fitz)` | Extract text from PDFs |
| `pdfplumber` | Extract tables from PDFs |
| `Pillow` | Handle images extracted from PDFs |
| `beautifulsoup4` | Scrape text, tables, images from web pages |
| `requests` | Fetch website content |
| `streamlit` | Web application front-end |
| `flask` | Website deployment (future integration) |
| `reportlab` | Generate downloadable PDFs |
| `pandas` | Handle extracted tables and dataframes |

To install required packages:
```bash
pip install torch transformers pymupdf pdfplumber pillow beautifulsoup4 requests streamlit flask reportlab pandas
```

## Instructions to Run
The streamlit application is already running on: http://magic02.cse.lehigh.edu:5005

To view the streamlit application, you need to first log into the Lehigh VPN.

1. **On the Lehigh VPN log into the magic 02 server.http://magic02.cse.lehigh.edu**:
2. **Load the streamlit app app.py (files are uploaded on the app)**:
3. **Install dependencies**:
   ```bash
   pip install torch transformers pymupdf pdfplumber pillow beautifulsoup4 requests streamlit flask reportlab pandas
   ```
   *(or manually install the packages listed above)*

4. **Identify the GPU with the lowest memory usage** asusing a GPU with a high memory usage can cause out of memory errors
5. **Launch the Streamlit app on terminal** Replace the CUDA_VISIBLE_DEVICE number "#" with the GPU number with the lowest memory usage:
   ```bash
   CUDA_VISIBLE_DEVICES= # streamlit run streamlitapp.py --server.port 5005
   ```
6. **Upload PDFs** or **enter a webpage URL** to generate a course module.
7. **Download** the generated lesson as a formatted PDF.

---

## Overview
This project develops an AI chatbot that generates structured course modules based on teacher-provided resources (PDFs, articles, web links) and example formats. The AI automates lesson creation, refines content based on teacher feedback, and dynamically updates a website with new modules.  
See the supporting documents for flow charts and code examples.

## Timeline (8 Weeks)
| Phase | Task | Duration | Deliverable |
|---|---|---|---|
| Week 1 | Model Selection & Setup | 1 week | LLM (Vicuna) running locally |
| Week 2 | PDF & Web Extraction | 1 week | Text extraction pipeline (PyMuPDF, BeautifulSoup) |
| Week 3-4 | AI-Powered Module Generation | 2 weeks | API for structured lesson generation |
| Week 5 | Website Development | 1 week | Website prototype (React/Next.js) |
| Week 6 | Integration & Auto-Updates | 1 week | AI-generated content deployment to website |
| Week 7 | Feedback & Refinement | 1 week | Teacher feedback system |
| Week 8 | Testing & Final Deployment | 1 week | Fully functional MVP |

## Value Over Commercial Models
- **Multi-Media Content Input**: Accepts PDFs, images, tables, and websites.
- **Multi-Media Content Output**: Produces web pages, PDFs, images, and tables.
- **Webpage Integration**: AI-generated modules populate a dynamic website.
- **Persistent Memory**: LLM retains teacher context across updates.

## Existing Solutions
### **Khanmigo by Khan Academy** [(Khan, 2024)](https://www.khanacademy.org/teacher/khanmigo-tools/lesson-plan?platform=KhanAcademy)
- **Benefit**: Creates lesson plans using Khan Academy’s internal content.
- **Limitation**: Does not allow external teacher inputs (e.g., PDFs or websites).

## Core Features
### **(a) Resource Processing**
- Extract text, images, and tables from PDFs and web pages.
- Utilize **PyMuPDF**, **PDFPlumber**, **Pillow**, **BeautifulSoup**.

### **(b) AI Module Generation**
- Generate structured lessons including topics, examples, and quizzes.
- Utilize **LLaVA** (multi-modal large language model).

### **(c) Website Integration**
- Deploy AI-generated modules to a dedicated website.
- Enable teacher feedback for continual refinement.

## Tech Stack
- **LLaMA**: Gen AI Model
- **BeautifulSoup**: HTML content extraction.
- **PyMuPDF**: PDF text and structure extraction.
- **PDFPlumber & Pillow**: PDF table and image extraction.
- **Flask**: Website backend for AI-generated content publishing.

---

## Contributors
- **Alex Kash**
- **Eddie McGowan**

---

## Documentation/References: Included in the documents in the documentation folder
