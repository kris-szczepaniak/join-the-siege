import pytest

from io import BytesIO
from src.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_no_file_in_request(client):
    response = client.post('/classify-file')
    assert response.status_code == 400

def test_no_selected_file(client):
    data = {'file': (BytesIO(b""), '')}  # Empty filename
    response = client.post('/classify-file', data=data, content_type='multipart/form-data')
    assert response.status_code == 400

def test_classification_failure(client, mocker):
    mocker.patch("src.app.validate_model_state", return_value=None)
    mocker.patch("src.app.get_and_validate_uploaded_file", return_value=BytesIO(b"dummy content"))
    mocker.patch("src.app.extract_text", return_value="dummy text")
    mocker.patch("src.app.validate_file_text", return_value=None)
    mocker.patch("src.app.classify_file", return_value=(None, None))
    
    data = {'file': (BytesIO(b"dummy content"), 'file.pdf')}
    response = client.post('/classify-file', data=data, content_type='application/pdf')
    assert response.status_code == 400
    assert response.get_json() == {"error": "Unable to classify document."}

def test_success(client, mocker):
    mocker.patch('src.app.validate_model_state', return_value=None)
    mocker.patch("src.app.get_and_validate_uploaded_file", return_value=BytesIO(b"foo bar"))
    mocker.patch("src.app.extract_text", return_value="foo bar")
    mocker.patch("src.app.validate_file_text", return_value=None)
    mocker.patch('src.app.classify_file', return_value=('test_class', 0.95))

    data = {'file': (BytesIO(b"dummy content"), 'file.pdf')}
    response = client.post('/classify-file', data=data, content_type='application/pdf')
    assert response.status_code == 200
    assert response.get_json() == {
        "file_class": "test_class", 
        "confidence": 0.95, 
        "file_text": "foo bar"
    }