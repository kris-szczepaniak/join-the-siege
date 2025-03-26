# Heron Coding Challenge - File Classifier

## Resources
1. **My model on Hugging Face**: https://huggingface.co/kris-szczepaniak/DistilBERT-document-classifier
2. **My dataset on Hugging Face (with explanations)**: https://huggingface.co/datasets/kris-szczepaniak/synthetic-documents-8k
3. **The code I used to generate synths and fine-tune the model**: https://github.com/kris-szczepaniak/heron-classifier-training


## Getting Started

1. Clone the repo
    ```
    git clone <repository_url>
    cd join-the-siege
    ```

2. Install dependencies
    ```
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Installing Tesseract OCR

    This project depends on Tesseract — you’ll need to install it separately before running the app.

    Ubuntu / Debian

        ```bash
        sudo apt update
        sudo apt install -y tesseract-ocr
        ```

    macOS

        ```bash
        brew update
        brew install tesseract
        ```

    Windows
    * follow instructions from here: https://github.com/tesseract-ocr/tessdoc

4. Run the app
    ```bash
    python run.py
    ```

5. Testing the classifier
    I used Postman to test this classifier, by issuing a `POST` request to `http://127.0.0.1:5001/classify-file`, and uploading the file by choosing 
    Body -> form-data -> key type: file -> uploading file

6. Running tests
    ```bash
    python -m pytest -p no:warnings
    ```

# Important Info

## Dataset
The model is fine-tuned on synthetic data, that I generated using methods described in my Hugging Face Dataset.

Why document categories include passport, driving license, contract and invoice? To satisfy the requirement of operability across sectors.

## What Is Supported
There's a lot of comments in the code, but broadly speaking:
* pdf, image in pdf,
* docx, image in docx,
* txt, 
* jpg, jpeg, png

## Dockerfile
I included a dummy dockerfile for deployment. I haven't tested it, but this is how I would start writing it if I had to deploy this app.

## Test Files
I added some test files into the `/files` folder. 

# Discussion

## The Goal
The goal is to preload the model on application startup, so that, given a high throughput of 100k req/day, a prediction is lightning fast. 100k files per day = 1.16 files/sec = 0.86 sec per file.

## Model Choice
My thoughts on the model choice revolved around topics such as:
* KNN Classifier
* Random Forest,
* Fine-Tuning a Pre-Trained Transformer Model,
* LLM,
* Computer Vision,
* RAG,

### RAG
Let's start with the last one. I thought it could be a nice and creative idea to use RAG for this task. Reading the specs again I decided agains it, as RAG is clunky and we don't necessarily need to converse with data. 

### Computer Vision Models
I decided against fine-tuning computer vision models (like Yolo v10), as they are too slow.

### KNN Classifier
For classic reasons, a KNN classifier was an obvious first choice. I trained a KNN Classifier, it worked fine, but had severe disadvantages:

* when I passed a random string it still provided a high (100%) prediction confidence,
* failed to understand a context -> as it shouldn't since it's not a transformer

### Random Forest
I thought about using Random Forest. I've used it in the past, had some limited experience with it. Alas, it doesn't solve the context problem.

### LLM
I decided against using a commercial LLM for this task, since that requires an additional request to a 3-rd party API, which is time and resource intensive, compared to loading a pre-trained model once. 

### [CHOSEN] **Fine-Tuning a Pre-Trained Transformer Model**

At first I thought I could fine-tune an old OpenAI's GPT model, but as far as I am aware, that sort of architecture would still require an additional API call on every classification request, as a fine-tuned model is hosted on OpenAI's servers. 

That's why I decided the best would be to fine-tune a smaller version of a battle-tested transformer model. I heard a thing or two about BERT, and that's how I found DistilBERT, Hugging Face's approach to reducing BERT to 66m params.

## Model Loading
After fine-tuning on synthetic data, I needed to export the model, so that I can use it in Heron Classifier.

This is the list of methods I though about and why I've chosen to go with the last one.

### Pickle
* Technically, any code could be executed on deserialization -> code injection attacks. Hence, deemed unfit.
* Not so portable outside of Python.

### Dumping Model Files Into Source Code
* Risk of repo bloating with model data.

### [CHOSEN] Hugging Face Model with Pre-Loading
* Avoids repo bloating
* Easy portability

## Limitations & Future Improvements

I am well aware of the areas in which the approach could be improved.

### Model Fit
A fairly limited document type set, cause the model to over-fit. On the other hand, dataset consisting of documents that need to comply with certain standards might be destined for over-fitting. 
That's why I am not including any training model metrics, as no matter the training params, the model overfits with the same stats. 
Nevertheless, it shouldn't be that big of an issue for now, given the fact that the documents that we want to classify always consist of the same sections, as previously mentioned.

### Training Limitations
* reads only english language,
* requires Human-In-The-Loop for misclassifications and low-confidence predictions (`pred < 0.9`)

### Code
* pydantic could be added to evaluate data models
* python-magic could be added to detect mime-type based on file content, not request-data
* batch file processing could be introduced
* more tests

## Conclusions
Full disclosure - that was the first time that I fine-tuned a transformer model on my own and posted it to Hugging Face.
All in all, it was a great experience for me and I am quite satisfied with the result. 

I hope you'll be too!
