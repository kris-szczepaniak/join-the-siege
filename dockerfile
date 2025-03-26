# NOT used, but a dockerfile would look something like this 

FROM python:3.12-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y tesseract-ocr

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Expose Flask’s default port
EXPOSE 8000

# Launch with Gunicorn binding to 0.0.0.0:8000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "run:app"]
