import pytest
import torch

from flask import Request
from io import BytesIO
from src.app import app
from src.settings.config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES
from src.utils.validators import get_and_validate_uploaded_file, is_allowed_file, validate_model_state, ValidationError
from unittest.mock import MagicMock
from werkzeug.datastructures import FileStorage

class MockFileStorage:
    def __init__(self, filename, mimetype, content=None):
        self.filename = filename
        self.mimetype = mimetype
        self.content = content

class MockRequest:
    def __init__(self, files):
        self.files = files

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.mark.parametrize("filename, ext, expected", [
    *[(f"file.{ext}", ext, True) for ext in ALLOWED_EXTENSIONS],
    ("file.gif", 'gif', False),
    ("file.svg", 'svg', False),
    ("file", '', False),
])
def test_allowed_file(filename, ext, expected):
    assert is_allowed_file(filename, ext) == expected

def test_model_state_valid():
    model = MagicMock()
    tokenizer = MagicMock()
    device = torch.device('cpu')

    validate_model_state(model, tokenizer, device)

@pytest.mark.parametrize("model, tokenizer, device", [
    (None, MagicMock(), torch.device("cpu")),
    (MagicMock(), None, torch.device("cuda")),
    (MagicMock(), MagicMock(), None),
    (None, None, None),
])
def test_model_state_invalid(model, tokenizer, device):
    with pytest.raises(ValidationError, match="Model or tokenizer is not loaded properly"):
        validate_model_state(model, tokenizer, device)

def test_no_file():
    with pytest.raises(ValidationError):
        get_and_validate_uploaded_file(MockRequest({}))


@pytest.mark.parametrize("filename, mimetype", [
    *[(None, val) for val in ALLOWED_MIME_TYPES.values()]
])
def test_empty_filename(filename, mimetype):
    with pytest.raises(ValidationError):
        get_and_validate_uploaded_file(MockRequest({MockFileStorage(filename, mimetype)}))

@pytest.mark.parametrize("filename, mimetype", [
    *[(f"test.{key}", None) for key in ALLOWED_MIME_TYPES]
])
def test_empty_mimetype(filename, mimetype):
    with pytest.raises(ValidationError):
        get_and_validate_uploaded_file(MockRequest({MockFileStorage(filename, mimetype)}))

@pytest.mark.parametrize("filename, mimetype", [
    *[(f"test.x{key}", val) for key, val in ALLOWED_MIME_TYPES.items()]
])
def test_filename_mimetype_mismatch(filename, mimetype):
    with pytest.raises(ValidationError):
        get_and_validate_uploaded_file(MockRequest({MockFileStorage(filename, mimetype)}))

@pytest.mark.parametrize("filename,mimetype,content", [
    ("test.txt", "text/plain", b"hello world"),
    ("UPPERCASE.TXT", "text/plain", b"HELLO"),
])
def test_success_returns_filestorage(client, filename, mimetype, content):
    request = Request.from_values(
        "/classify-file", method="POST",
        data={"file": (BytesIO(content), filename, mimetype)},
    )
    file_obj = get_and_validate_uploaded_file(request)

    assert isinstance(file_obj, FileStorage)
    assert file_obj.filename == filename
    assert file_obj.mimetype == mimetype