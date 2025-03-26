ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt'}

ALLOWED_MIME_TYPES = {
    'pdf': 'application/pdf',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'txt': 'text/plain'
}

ALLOWED_IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png')

ID_TO_LABEL = {0: 'invoice', 1: 'driving_license', 2: 'contract', 3: 'passport'}