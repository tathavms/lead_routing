# ML-Based Customer Support Ticket Router

> Classifies incoming support tickets into departments using a fine-tuned DistilBERT model, with a transformer-based spam filter as the first line of defence. Fully containerised and deployed on AWS EC2 behind Nginx.

**Live demo:** `http://13.48.162.241/lead-routing/`

---

## What it does

A user submits a support query. The system:
1. Checks if it's spam (transformer classifier)
2. If not, routes it to the correct department (fine-tuned DistilBERT)
3. Rejects low-confidence or out-of-scope inputs rather than guessing wrong

```
User Input
    │
    ▼
┌─────────────────────┐
│   Spam Classifier   │  ──► SPAM → Blocked
│  (tanaos/spam-v1)   │
└─────────────────────┘
    │ HAM
    ▼
┌─────────────────────┐
│  Confidence Check   │  ──► < 60% or high entropy → "Unrecognised"
│  + Entropy Filter   │
└─────────────────────┘
    │ PASS
    ▼
┌─────────────────────┐
│   DistilBERT        │
│  Department Router  │  ──► Tech Support / Customer Service / Billing / Sales
└─────────────────────┘
```

**Departments:**

| Label | Maps from |
|---|---|
| Tech Support | Technical Support, IT Support, Service Outages |
| Customer Service | Customer Service, Returns, Product Support |
| Billing | Billing & Payments, General Inquiry |
| Sales | Sales & Pre-Sales |

---

## Architecture

```
                        ┌──────────────┐
                        │   Browser    │
                        └──────┬───────┘
                               │ HTTP :80
                        ┌──────▼───────┐
                        │    Nginx     │  reverse proxy
                        └──────┬───────┘
               ┌───────────────┴────────────────┐
               │ /lead-routing/                 │ /twitter-sentiment-analysis/
        ┌──────▼───────┐                 ┌───────▼──────┐
        │  FastAPI     │                 │  FastAPI     │
        │  :8000       │                 │  :8001       │
        │  (Docker)    │                 │  (Docker)    │
        └──────┬───────┘                 └──────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌─────▼──────┐
│  Spam  │          │ DistilBERT │
│ Model  │          │  Router    │
└────────┘          └─────┬──────┘
                          │ weights via S3 volume mount
                    ┌─────▼──────┐
                    │  AWS S3    │
                    │ (LFS store)│
                    └────────────┘
```

**EC2 t2.micro** — 1GB RAM + 2GB swap. Two ML apps coexisting via Nginx routing.

---

## CI/CD Pipeline

Two independent GitHub Actions workflows, each with a single responsibility:

```
Code change pushed
        │
        ▼
deploy_image.yml
  ├── Build Docker image (no model weights — keeps image lean)
  ├── Push to GHCR
  └── SSH → EC2: pull image, restart container with S3-mounted weights

Model weights changed (LFS)
        │
        ▼
deploy_s3.yml
  ├── Checkout with LFS (actual bytes, not pointer)
  ├── Upload both .safetensors to S3
  └── SSH → EC2: download weights to host, restart container
```

Model weights live on the EC2 host disk (pulled from S3) and are volume-mounted into the container at runtime. The Docker image itself stays under 1GB.

---

## ML Pipeline

### Data Preparation (`01_data_prep.ipynb`)
- Loaded raw business support email dataset
- Merged 10 granular labels into 4 business-meaningful departments
- Dropped HR tickets (internal employee queries — out of scope for customer support routing)
- Checked class distribution and flagged imbalance for downstream handling

### Spam Detection (`02_spam_detection.ipynb`)
- Evaluated 13 classifiers (SVC, Naive Bayes, Random Forest, XGBoost, etc.)
- Selected by precision over accuracy — false negatives (spam getting through) are more costly than false positives
- Pre-trained transformer (`tanaos/tanaos-spam-detection-v1`) used in production for domain generalisability

### Department Classifier (`03_train_and_evaluate.ipynb`)
- Fine-tuned `distilbert-base-uncased` on ~20k labelled support tickets
- Fixed non-deterministic label mapping (`sorted()` before encoding) that was causing silent accuracy degradation across runs
- Injected short-query synthetic examples (split before duplication to prevent data leakage into eval set)
- Stratified train/test split to handle class imbalance
- `WeightedTrainer` subclass with inverse-frequency class weights for underrepresented departments
- Dual OOD rejection: confidence threshold (60%) + entropy check (70% of max entropy for 4 classes)

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model training | PyTorch, HuggingFace Transformers, Datasets |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Containerisation | Docker (multi-stage build) |
| Reverse proxy | Nginx |
| Cloud | AWS EC2 (t2.micro) |
| Model storage | AWS S3 (large file store, out of Docker image) |
| Registry | GitHub Container Registry (GHCR) |
| CI/CD | GitHub Actions (two-workflow pattern) |
| Version control (LFS) | Git LFS for `.safetensors` files |

---

## Project Structure

```
lead-routing-customer-support/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, static file serving
│   ├── predict.py       # Spam check + department routing logic
│   └── schemas.py       # Pydantic request/response models
├── frontend/
│   ├── index.html
|   ├── style.css
│   └── script.js
├── models/
│   ├── routing_model/   # config, tokenizer (weights via S3)
│   └── spam_detection_model/
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_spam_detection.ipynb
│   └── 03_train_and_evaluate.ipynb
├── .github/workflows/
│   ├── deploy_image.yml   # builds and deploys container
│   └── deploy_s3.yml      # syncs model weights to S3 + EC2
└── Dockerfile             # multi-stage build
```

---

## API

**POST** `/lead-routing/predict`

```json
// Request
{ "user_input": "My invoice has incorrect charges from last month" }

// Response — routed
{ "is_spam": false, "department": "Billing", "confidence": 94.3 }

// Response — spam
{ "is_spam": true, "department": null, "confidence": null }

// Response — out of scope
{ "is_spam": false, "department": "unrecognised", "confidence": 38.1 }
```

**GET** `/lead-routing/health` → `{ "status": "ok" }`

Interactive docs: `http://13.48.162.241/lead-routing/docs`

---

## Inspired by

An internal ML Lead Routing system, which reduced average SQL response time by 57% using BERT-based classification across Sales and Support queues. This project replicates that architecture for a customer support context using open-source tooling.
