import torch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from typing import Tuple

"""
The goal is to preload the model on application startup, so that, given a 
high throughput of 100k req/day, a prediction is lightning fast. 

Target? 1.16 files/sec, so total request time per file <= 0.86 sec.

I thought about a couple of methods for using my model here. I fine-tuned it
on synthetic data, using other code, that I put in a separate repo and will share
with you separately.

Let's consider our options.

1. Pickle
    - Probably not a good idea for production applications, since loading from pickle
    isn't considered the safest method. Technically, any code could be executed
    while deserialization -> code injection attacks. Hence deemed out of scope.
    - Not so portable outside of python

2. Dumping model files into source code
    - Bloating the repo with 280MB of model data in the source code... not the greatest thing to do.
    Deemed out of scope

3. Hugging Face download
    - Still needs to load the model from HF repo, still 280MB, but way safer,
    neater and best out of the 3 for production apps. It's also cross-platform.
    - Avoids repo bloating
    - Easy portability
"""

""" 
    My thoughts on the model choice oscillated around 3 main topics:
        - KNN classifier, just for classic reasons,
        - Random Forest / Gradient Boosting,
        - fine-tuning a pre-trained transformer model,
        - Computer Vision (:O),
        - RAG (:O). 

    1. RAG
        Let's start with the last one. I thought it could be a nice and creative idea to use RAG for this task.
        But then I read the specs again and though - 100k/day throughput - it sounds like RAG could be an overkill here,
        as it is clunky and I don't necessarily need to converse with data, nor do I plan to upload the entire contracts or
        books here. So that idea I dropped. 

    2. KNN Classifier
        For classic reasons, a KNN classifier was an obvious choice, I thought. First thought - I am gonna find a nice 
        training dataset online. I couldn't. So I decided to create a set of over 8k synthetic OCR-like reads from a 
        couple of commonly used document types. How I did that is described in my Hugging Face repo, that I am going to 
        link in the submission. 
        
        Knowing that we have a high throughput case here, I decided against fine-tuning computer vision models (like Yolo v10), 
        as they are too slooooow.

        That's why I decided, that simplicity is my best friend and simulated text read from a file, by any means necessary,
        and labeled it (code in another repo, indicated in submission as well). 

        I trained a KNN Classifier, it worked fine, but had severe disadvantages:
            - when I passed a random string it still provided a high (100%) prediction confidence (duh it did),
            - failed to understand a context -> as it shouldn't since it's not a transformer :)

        Pivot was needed.

    3. Random Forest
        I thought about using Random Forest. I've used it in the past, had some limited experience with it.
        Alas, I dropped this idea very fast, for the same reasons I dropped KNN - it doesn't understand a context
        and is very sensitive to token shuffling. 

        Pivot x2. 

    4. LLM + Fine-tuning a pre-trained model
        At first I thought I could do it with fine-tuning an old OpenAI's GPT model, but then again, that sort
        of architecture would require additional API call on every classification request. That requires 1) time, that 
        we don't have since our throughput is so high, and 2) network, which generally is unreliable. 

        I thought the only shot at this that I am aware of, is to fine-tune a smaller version of a battle-tested
        transformer model. I knew a thing or two about BERT, and that's how I found DistilBERT, Hugging Face's 
        approach to reducing the model size to 66m params. 

    I am well aware of the areas in which it could have been improved, such as: 
        - a fairly limited document type set, 
        - reads only english language,
        - requires Human-In-The-Loop for misclassifications and low-confidence predictions,
        - pydantic could be added to evaluate data models
        - python-magic could be added to detect mime-type based on file content, not request-data
        - batch file processing could be introduced
    
    Full disclosure - that was the first time that I fine-tuned a transformer model on my own and posted it to 
    Hugging Face. 
    
    All in all, it was a great experience for me and I am quite satisfied with the result. I hope you'll be too!
"""

def load_model_and_tokenizer(model_name: str = 'kris-szczepaniak/DistilBERT-document-classifier') -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizer, torch.device]:
    """
    Loads DistilBERT-type model for sequence classification and a corresponding tokenizer from HuggingFace repository.

    :param model_name: must be a DistilBERT Sequence Classifier
    :type model_name: str
    :return: A tuple consisting of: pretrained model instance, tokenizer instance and detected device type
    :rtype: Tuple[DistilBertForSequenceClassification, DistilBertTokenizer, torch.device]
    """
    if '/' not in model_name or len(model_name) < 10:
        raise ValueError("model_name incorrect")
    
    try:
        pretrained_model = DistilBertForSequenceClassification.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        device = torch('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pretrained_model.to(device)
        
        print(f"Model {model_name} loaded successfully!")

        return pretrained_model, tokenizer, device
    
    except OSError as e:
        print(f"Failed to load model/tokenizer: {e}")
    except RuntimeError as e:
        print(f"Runtime error (most likely CUDA issues): {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None, None, None


