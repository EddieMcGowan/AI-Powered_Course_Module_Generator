from flask import Flask, render_template, request, send_from_directory
import os
from model_utils import process_doc, reprompt_lesson, render_output_with_images_and_tables

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create the upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Route to serve images from the images folder
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

# Temporary in-memory state (good enough for single-user dev mode)
CURRENT_LESSON = ""
CURRENT_TABLES = []
CURRENT_IMAGES = []

@app.route("/", methods=["GET", "POST"])
def index():
    global CURRENT_LESSON, CURRENT_TABLES, CURRENT_IMAGES

    lesson_display = ""
    message = ""

    if request.method == "POST":
        # PDF Upload
        if "pdf" in request.files:
            uploaded_file = request.files["pdf"]
            if uploaded_file.filename != "":
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(filepath)

                CURRENT_LESSON, lesson_display, CURRENT_TABLES, CURRENT_IMAGES = process_doc(filepath)

        # Teacher Instruction
        elif "instruction" in request.form:
            instruction = request.form["instruction"]
            if instruction.strip().lower() == "approved":
                message = "✅ Lesson approved by the teacher."
                lesson_display = render_output_with_images_and_tables(CURRENT_LESSON, CURRENT_IMAGES, CURRENT_TABLES)
            else:
                CURRENT_LESSON = reprompt_lesson(CURRENT_LESSON, instruction)
                lesson_display = render_output_with_images_and_tables(CURRENT_LESSON, CURRENT_IMAGES, CURRENT_TABLES)

    return render_template("index.html", lesson=lesson_display, message=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
