import torch
import torch.nn.functional as F

from src.settings.config import ID_TO_LABEL
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from typing import Dict, Tuple


def prepare_text(text: str, tokenizer: DistilBertTokenizer, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    This function is to prepare the raw text, so that it can be passed to a transformer model by 
    tokenizing it and converting to PyTorch tensors.
    
    :param text: - raw input string to be tokenized
    :type text: str
    :param tokenizer: - DistilBertTokenizer-type; converts raw text to token IDs, attention masks, and all that a transformer model needs
    :type tokenizer: transformers.DistilBertTokenizer
    :param device: - CPU or GPU, inferred by PyTorch while preloading the model
    :type device: torch.device
    :return: Dictionary of tokenized input tensors ready for the model.
    :rtype: Dict[str, torch.Tensor]
    """
    encoded_input = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    return encoded_input.to(device)


def classify_text(model: DistilBertForSequenceClassification, prepared_text: Dict[str, torch.Tensor], id_to_label: Dict[int, str] = ID_TO_LABEL) -> Tuple[str, float]:
    """
    Classifies text using a transformer model.

    :param model: The transformer model to use for classification.
    :type model: transformers.DistilBertForSequenceClassification
    :param prepared_text: The tokenized and prepared input text.
    :type prepared_text: Dict[str, torch.Tensor]
    :param id_to_label: A dictionary mapping class IDs to labels.
    :type id_to_label: Dict[int, str]
    :return: The predicted label and confidence score.
    :rtype: Tuple[str, float]
    """
    model.eval() # we don't need to train, evaluation mode enabled
    with torch.no_grad(): # disable gradient calculation for speed
        outputs = model(**prepared_text)
        logits = outputs.logits # get raw output
        probabilities = F.softmax(logits, dim=1) # softmax to get probs
        predicted_class_id = torch.argmax(probabilities, dim=1).item() # get the class with the highest pred. prob
        predicted_label = ID_TO_LABEL[predicted_class_id]
        confidence = probabilities[0, predicted_class_id].item() # pred confidence score

    return predicted_label, confidence
