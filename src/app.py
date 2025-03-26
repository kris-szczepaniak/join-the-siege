from flask import Flask, request, jsonify

from src.model.model_preloader import load_model_and_tokenizer

from src.utils.classifier import classify_file
from src.utils.error_interceptor import error_interceptor
from src.utils.text_extractor import extract_text
from src.utils.validators import validate_model_state, get_and_validate_uploaded_file, validate_file_text

app = Flask(__name__)

pretrained_model, tokenizer, device = load_model_and_tokenizer()

@app.route('/classify-file', methods=['POST'])
@error_interceptor
def classify_file_route():
    """
    Classifies a file uploaded via POST request.

    :return: JSON response with classification results or error message.
    """
    # could be done with pydantic as well
    validate_model_state(pretrained_model, tokenizer, device)

    file = get_and_validate_uploaded_file(request)

    file_text = extract_text(file)
    validate_file_text(file_text)
    
    file_class, confidence = classify_file(file_text, pretrained_model, tokenizer=tokenizer, device=device)
    
    if all([file_text, file_class, confidence]):
        return jsonify({"file_class": file_class, "confidence": confidence, "file_text": file_text}), 200
    
    return jsonify({"error": "Unable to classify document."}), 400

