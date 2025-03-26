import torch

from src.model.model_utils import classify_text, prepare_text
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from typing import Tuple


def classify_file(text: str, model: DistilBertForSequenceClassification, tokenizer: DistilBertTokenizer, device: torch.device) -> Tuple[str, float]:
    """
    Classifies raw file text data using model and tokenizer of DistilBERT-type provided.

    :param text: Raw text to classify
    :type text: str
    :param model: DistilBERT-type model instance
    :type model: transformers.DistilBertForSequenceClassification
    :param tokenizer: DistilBERT-type tokenizer, applied to text
    :type tokenizer: transformers.DistilBertTokenizer
    :param device: Device type (CPU or GPU) as read by PyTorch
    :type device: torch.device
    :return: A tuple consisting of a predicted label (string) and associated confidence in range 0-1 (float).
    :rtype: Tuple[str, float]
    """

    if text is not None and len(text.strip()) > 0:
        prepared_text = prepare_text(text, tokenizer, device)
        predicted_label, confidence = classify_text(model, prepared_text)

        return predicted_label, confidence
    else:
        return None, None

