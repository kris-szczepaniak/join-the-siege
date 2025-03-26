import pytest
from io import BytesIO
from werkzeug.datastructures import FileStorage

from src.utils.text_extractor import (
    extract_file_extension,
    extract_text_from_txt,
    extract_text,
)

@pytest.mark.parametrize("filename,expected", [
    ("foo.txt", "txt"),
    ("BAR.PDF", "pdf"),
    ("multi.part.name.docx", "docx"),
])
def test_extract_file_extension(filename, expected):
    assert extract_file_extension(filename) == expected

def test_extract_text_from_txt_valid_utf8():
    data = "Hello world\nLine 2".encode("utf-8")
    assert extract_text_from_txt(data) == "Hello world\nLine 2"

def test_extract_text_from_txt_invalid_utf8(capfd):
    bad_bytes = b"\xff\xfe\xfa"
    result = extract_text_from_txt(bad_bytes)
    captured = capfd.readouterr()
    assert result == ""
    assert "File encoding is not valid UTF-8." in captured.out

def make_filestorage(content: bytes, filename: str, mimetype: str="text/plain"):
    stream = BytesIO(content)
    return FileStorage(stream=stream, filename=filename, content_type=mimetype)

def test_extract_text_txt_file():
    fs = make_filestorage(b"some text", "example.txt")
    assert extract_text(fs) == "some text"

def test_extract_text_unsupported_extension(capfd):
    fs = make_filestorage(b"", "file.xyz")
    assert extract_text(fs) is None
    captured = capfd.readouterr()
    assert "Error: Unsupported file type" in captured.out