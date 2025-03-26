import torch

from flask import Request

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from src.settings.config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES
from src.utils.text_extractor import extract_file_extension

from werkzeug.datastructures import FileStorage


class ValidationError(Exception):
    def __init__(self, message):
        super().__init__(message)

def is_allowed_file(filename: str, ext: str) -> bool:
    """
    Validates whether the file is allowed by checking filename and extension.

    :param filename:
    :type filename: str
    :param ext: file extension
    :type ext: str
    :return: is filed allowed
    :rtype: bool
    """
    return '.' in filename and ext in ALLOWED_EXTENSIONS

def validate_model_state(pretrained_model: DistilBertForSequenceClassification, tokenizer: DistilBertTokenizer, device: torch.device):
    """
    Checks if model has been preloaded correctly

    :param pretrained_model: DistilBERT-type model instance
    :type pretrained_model: transformers.DistilBertForSequenceClassification
    :param tokenizer: DistilBERT-type tokenizer, applied to text
    :type tokenizer: transformers.DistilBertTokenizer
    :param device: Device type (CPU or GPU) as read by PyTorch
    :type device: torch.device
    """
    if not all([pretrained_model, tokenizer, device]):
        raise ValidationError("Model or tokenizer is not loaded properly. Ensure the model name is correct or try again later.")
    
def get_and_validate_uploaded_file(request: Request) -> FileStorage:
    """
    Validates file received in the request

    :param request: Received request object
    :type request: flask.Request
    :return: validate and extracted file
    :rtype: FileStorage
    """
    if 'file' not in request.files:
        raise ValidationError("No file provided")

    file = request.files['file']
    
    if file.filename == '':
        raise ValidationError("Empty filename")

    ext = extract_file_extension(file.filename)

    if not is_allowed_file(file.filename, ext):
        raise ValidationError(f"File type not allowed. Allowed types include: {', '.join(ALLOWED_EXTENSIONS)}")
    
    expected_mime = ALLOWED_MIME_TYPES.get(ext)

    if file.mimetype != expected_mime: # python-magic would be a nice to have to check real mime-type of the file content
        raise ValidationError(f"MIME type doesn't match expected type for .{ext}. Expected: {expected_mime}, got: {file.mimetype}")
    
    return file

def validate_file_text(file_text: str):
    """
    Validates whether extracted text is valid

    :param file_text: Extracted text
    :type file_text: str
    """
    if not file_text or len(file_text.strip()) == 0:
        raise ValidationError("Text extraction result is empty")