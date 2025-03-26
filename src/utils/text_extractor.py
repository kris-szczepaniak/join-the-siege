import pymupdf
import pytesseract

from docx import Document
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError
from src.settings.config import ALLOWED_IMAGE_EXTENSIONS
from werkzeug.datastructures import FileStorage

def extract_file_extension(filename: str) -> str:
    """
    Extracts file extension from filename

    :param filename: Filename
    :type filename: str
    :return: Extension extracted
    :rtype: str
    """
    return filename.rsplit('.', 1)[1].lower()

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extracts text from a byte image representation.
    Increases sharpness and contrast for better reading.

    :param image_bytes: byte representation of an image
    :image_bytes type: bytes
    :return: Text read from the image
    :rtype: str
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert('L')
        image = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1)

        text = pytesseract.image_to_string(image)
        return text.strip()
    
    except UnidentifiedImageError:
        print("Error: Invalid image data")
        return ""
    except pytesseract.TesseractError:
        print("Error: Tesseract failed to extract text")
        return ""
    
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from a byte pdf representation. 
    If no text can be found, function tries to convert
    file_bytes to image and supply to extract_text_from_image(). 

    :param file_bytes: byte representation of the PDF file
    :file_bytes type: bytes
    :return: Text read from the PDF
    :rtype: str
    """
    try:
        pdf_doc = pymupdf.open(stream=file_bytes, filetype='pdf')

        text = ''
        for page_num in range(len(pdf_doc)):
            try:
                page = pdf_doc.load_page(page_num)

                page_text = page.get_text().strip()

                if page_text:
                    text += page_text + '\n'

                else:
                    pix_map = page.get_pixmap(dpi=300)
                    image_bytes = pix_map.tobytes("png")
                    image_text = extract_text_from_image(image_bytes)

                    if image_text:
                        text += image_text + '\n'
            
            except pymupdf.FileDataError as e:
                print(f"Structural error occured while processing page {page_num}: {e}")

        pdf_doc.close()
        return text.strip()
    
    except pymupdf.FileDataError as e:
        print(f"Error reading file bytes: {e}")
        return ""
    
def extract_text_from_docx(docx_bytes: bytes) -> str:
    """
    Extracts text from a byte representation of docx.
    If no text can be found, function tries to convert
    docx_bytes to image and supply to extract_text_from_image().

    :param docx_bytes: byte representation of the DOCX file
    :docx_bytes type: bytes
    :return: Text read from the DOCX
    :rtype: str
    """
    try:
        document = Document(BytesIO(docx_bytes))

        text = '\n'.join([p.text for p in document.paragraphs])
        
        if not text.strip():
            image_texts = []
            
            for rel in document.part.rels.values():
                if "image" in rel.reltype:
                    try:
                        image_data = rel.target_part.blob
                        text = extract_text_from_image(image_data)
                        image_texts.append(text)
                    
                    except Exception as e:
                        print(f"Error extracting text from image in docx: {e}")
            
            text = "\n".join(image_texts)
        
        return text.strip()
    
    except Exception as e:
        print(f"Error processing docx: {e}")
        return ""
    
def extract_text_from_txt(txt_bytes: bytes) -> str:
    """
    Extracts text from a byte representation of txt.

    :param txt_bytes: byte representation of the TXT file
    :txt_bytes type: bytes
    :return: Text read from the TXT
    :rtype: str
    """
    try:
        text = txt_bytes.decode('utf-8')
        return text.strip()
    
    except UnicodeDecodeError:
        print("File encoding is not valid UTF-8.")
        return ""

def extract_text(file: FileStorage) -> str:
    """
    Governs text extraction from file. 

    :param file: File received in request form-data
    :type file: FileStorage
    :return: Text read from the file
    :rtype: str
    """

    filename = file.filename.lower().strip()
    
    if filename.endswith('pdf'):
        file_bytes = file.read()
        text = extract_text_from_pdf(file_bytes)
        
        return text
    
    elif filename.endswith('.docx'):
        docx_bytes = file.read()
        text = extract_text_from_docx(docx_bytes)

        return text
    
    elif filename.endswith('.txt'):
        txt_bytes = file.read()
        text = extract_text_from_txt(txt_bytes)

        return text
        
    elif filename.endswith(ALLOWED_IMAGE_EXTENSIONS):
        image_bytes = file.read()
        text = extract_text_from_image(image_bytes)
        
        return text
    
    else:
        print("Error: Unsupported file type")
        return None