# AI-Powered Course Module Generator

## Overview
This project develops an AI chatbot that generates structured course modules based on teacher-provided resources (PDFs, articles, web links) and example formats. The AI automates lesson creation, refines content based on teacher feedback, and dynamically updates a website with new modules.

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
- **Multi-Media Content Input**: Unlike free ChatGPT, this system will accept PDFs, images, and tables as inputs.
- **Multi-Media Content Output**: Outputs can include web pages, PDFs, images, or tables.
- **Webpage Integration**: AI-generated responses will populate a dedicated website.
- **Persistent Memory**: The LLM will retain context to reduce iteration time for teachers.

## Existing Solutions
### **Khanmigo by Khan Academy** [(Khan, 2024)](https://www.khanacademy.org/teacher/khanmigo-tools/lesson-plan?platform=KhanAcademy)
**Benefit**: Creates lesson plans from its own content.
**Limitation**: Does not allow teachers to input specific sources.

## Core Features
### **(a) Resource Processing**
- Extract text, images, and tables from PDFs, articles, and web pages.
- Utilize **PyMuPDF, PDFPlumber, Pillow, BeautifulSoup** for extraction.

### **(b) AI Module Generation**
- Generate structured lessons including topics, examples, and Q&A sections.
- Utilize **LLaVA** for generation.

### **(c) Website Integration**
- Deploy modules automatically.
- Enable teacher feedback and updates.

## Tech Stack
- **LLaVA**: Pretrained model supporting text, images, and tables as input.
- **BeautifulSoup**: Extract images, text, and tables from web pages.
- **PyMuPDF**: Extract text from PDFs.
- **PDFPlumber & Pillow**: Extract images and tables from PDFs.
- **Flask**: Publish LLM responses to the website.

## Next Steps
- Set up **LLaVA** model on an **HCP** and adjust based on configuration.
- Build **text extraction pipeline** using PyMuPDF & BeautifulSoup.
- Develop **AI-generated module API**.
- Deploy **website & feedback system**.

---
### Contributors
- **Alex Kash**
- **Eddie McGowan**
