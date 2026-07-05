import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# loading spam model artifacts
spam_config    = AutoConfig.from_pretrained('../models/spam_detection_model')
spam_tokenizer = AutoTokenizer.from_pretrained('../models/spam_detection_model')
spam_model     = AutoModelForSequenceClassification.from_pretrained('../models/spam_detection_model')
spam_model.eval()

# loading routing model artifacts
with open(f'../models/routing_model/queue_mapping.pkl', 'rb') as f:
    queue_mapping=pickle.load(f)

id_to_queue = {v: k for k, v in queue_mapping.items()}
tokenizer      = AutoTokenizer.from_pretrained('../models/routing_model')
routing_model  = AutoModelForSequenceClassification.from_pretrained('../models/routing_model')
routing_model.eval()

# checking if input is spam using transformer model
def is_spam(text: str) -> bool:
    inputs = spam_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs.pop('token_type_ids', None)
    with torch.no_grad():
        outputs = spam_model(**inputs)
    pred_idx = torch.argmax(outputs.logits, dim=1)[0].item()
    return spam_config.id2label[pred_idx] == 'spam'

# computing entropy of prediction — high entropy means model is unsure
def get_entropy(logits: torch.Tensor) -> float:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # entropy formula: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
    return entropy

CONFIDENCE_THRESHOLD = 60.0
# routing to department using distilbert
def predict_department(text: str) -> tuple[str | None, float | None]:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    # removing token_type_ids — distilbert doesn't accept this argument
    inputs.pop('token_type_ids', None)

    with torch.no_grad():
        outputs = routing_model(**inputs)

    entropy = get_entropy(outputs.logits)   
    pred_idx   = torch.argmax(outputs.logits, dim=-1).item()
    confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][pred_idx].item()
    confidence_pct = round(confidence * 100, 2)
    
    # max entropy for 4 classes is log(4) ≈ 1.386 — threshold at 70% of max
    MAX_ENTROPY = torch.log(torch.tensor(4.0)).item()
    ENTROPY_THRESHOLD = MAX_ENTROPY * 0.70

    # rejecting low confidence OR high entropy predictions as irrelevant input
    if confidence_pct < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        return None, confidence_pct

    return id_to_queue[pred_idx], confidence_pct
