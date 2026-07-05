# import pickle
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
# from pathlib import Path

# CURRENT_DIR = Path(__file__).resolve().parent
# MODEL_PATH = CURRENT_DIR / ".." / "models"

# # loading spam model artifacts
# spam_config    = AutoConfig.from_pretrained(MODEL_PATH / 'spam_detection_model')
# spam_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH / 'spam_detection_model')
# spam_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH / 'spam_detection_model')
# spam_model.eval()

# # loading routing model artifacts
# with open(MODEL_PATH / 'routing_model/queue_mapping.pkl', 'rb') as f:
#     queue_mapping=pickle.load(f)

# id_to_queue = {v: k for k, v in queue_mapping.items()}
# tokenizer      = AutoTokenizer.from_pretrained(MODEL_PATH / 'routing_model')
# routing_model  = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH / 'routing_model')
# routing_model.eval()

# # checking if input is spam using transformer model
# def is_spam(text: str) -> bool:
#     inputs = spam_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     inputs.pop('token_type_ids', None)
#     with torch.no_grad():
#         outputs = spam_model(**inputs)
#     pred_idx = torch.argmax(outputs.logits, dim=1)[0].item()
#     return spam_config.id2label[pred_idx] == 'spam'

# # computing entropy of prediction — high entropy means model is unsure
# def get_entropy(logits: torch.Tensor) -> float:
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     # entropy formula: -sum(p * log(p))
#     entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
#     return entropy

# CONFIDENCE_THRESHOLD = 60.0
# # routing to department using distilbert
# def predict_department(text: str) -> tuple[str | None, float | None]:
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
#     # removing token_type_ids — distilbert doesn't accept this argument
#     inputs.pop('token_type_ids', None)

#     with torch.no_grad():
#         outputs = routing_model(**inputs)

#     entropy = get_entropy(outputs.logits)   
#     pred_idx   = torch.argmax(outputs.logits, dim=-1).item()
#     confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][pred_idx].item()
#     confidence_pct = round(confidence * 100, 2)
    
#     # max entropy for 4 classes is log(4) ≈ 1.386 — threshold at 70% of max
#     MAX_ENTROPY = torch.log(torch.tensor(4.0)).item()
#     ENTROPY_THRESHOLD = MAX_ENTROPY * 0.70

#     # rejecting low confidence OR high entropy predictions as irrelevant input
#     if confidence_pct < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
#         return None, confidence_pct

#     return id_to_queue[pred_idx], confidence_pct

import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / ".." / "models"

# declaring as None — loading on first use instead of at startup
spam_config = None
spam_tokenizer = None
spam_model = None
queue_mapping = None
id_to_queue = None
routing_tokenizer = None
routing_model = None

def load_models():
    global spam_config, spam_tokenizer, spam_model
    global queue_mapping, id_to_queue, routing_tokenizer, routing_model

    # only loading if not already loaded
    if spam_model is not None:
        return

    spam_config    = AutoConfig.from_pretrained(MODEL_PATH / 'spam_detection_model')
    spam_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH / 'spam_detection_model')
    spam_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH / 'spam_detection_model')
    spam_model.eval()

    with open(MODEL_PATH / 'routing_model/queue_mapping.pkl', 'rb') as f:
        queue_mapping = pickle.load(f)

    id_to_queue = {v: k for k, v in queue_mapping.items()}
    routing_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH / 'routing_model')
    routing_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH / 'routing_model')
    routing_model.eval()

def is_spam(text: str) -> bool:
    load_models()
    inputs = spam_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs.pop('token_type_ids', None)
    with torch.no_grad():
        outputs = spam_model(**inputs)
    pred_idx = torch.argmax(outputs.logits, dim=1)[0].item()
    return spam_config.id2label[pred_idx] == 'spam'

def get_entropy(logits: torch.Tensor) -> float:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
    return entropy

CONFIDENCE_THRESHOLD = 60.0

def predict_department(text: str) -> tuple[str | None, float | None]:
    load_models()
    inputs = routing_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs.pop('token_type_ids', None)

    with torch.no_grad():
        outputs = routing_model(**inputs)

    entropy = get_entropy(outputs.logits)
    pred_idx   = torch.argmax(outputs.logits, dim=-1).item()
    confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][pred_idx].item()
    confidence_pct = round(confidence * 100, 2)

    MAX_ENTROPY = torch.log(torch.tensor(4.0)).item()
    ENTROPY_THRESHOLD = MAX_ENTROPY * 0.70

    if confidence_pct < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        return None, confidence_pct

    return id_to_queue[pred_idx], confidence_pct