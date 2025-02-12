import re
from typing import Dict
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def clean_text(text:str) -> str:
        """
        This method cleans prompt data, removing extraneous punctuation meant to denote blending, loras, or models without removing names or tags. 
        We also get rid of extraneous spaces or line breaks to reduce tokens and maintain as much semantic logic as possible
        """
        text = str(text)
        # Remove additional characters: ( ) : < > [ ]
        cleaned_text = re.sub(r'[():<>[\]]', ' ', text)
        cleaned_text = cleaned_text.replace('\n', ' ')
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s*,\s*', ', ', cleaned_text)

        return cleaned_text.strip()

def tokenize_function(examples: Dict[str,str], column: str, tokenizer:AutoTokenizer, max_length: int =512):
    # Convert 'text' column to string
    examples['text'] = [str(text) for text in examples[column]]
    return tokenizer(examples['text'], max_length=max_length, truncation=True, padding='max_length')


# Define your metric function
def compute_bert_metrics(p, metric_name):
    labels = p.label_ids
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }