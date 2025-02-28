from flask import Flask, request, jsonify
from transformers import pipeline
import PyPDF2
import pdfplumber
import io

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model for text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased")

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    text = ""
    try:
        # Try using PyPDF2
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extract_text()
    except:
        # Fallback to pdfplumber
        file.seek(0)  # Reset file pointer
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    return text

# Function to extract text from a .txt file
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Route to handle file uploads and check vulgarity
@app.route('/check_vulgarity', methods=['POST'])
def check_vulgarity():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Extract text based on file type
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith('.txt'):
            text = extract_text_from_txt(file)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        # Check if text is extracted
        if not text:
            return jsonify({'error': 'No text extracted from the file'}), 400

        # Use the model to classify the text
        result = classifier(text)[0]

        # Determine if the text is acceptable
        is_acceptable = result['label'] == 'LABEL_0'

        # Return the result
        return jsonify({
            'is_acceptable': is_acceptable,
            'confidence': result['score'],
            'label': result['label'],
            'extracted_text': text[:500]  # Return first 500 characters for debugging
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)