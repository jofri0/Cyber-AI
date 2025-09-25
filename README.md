# Cyber AI Project v2

This is an extended starter project for building an AI specialized in programming,
cybersecurity, and computer/internet-related topics.

## Features
- Modular design for different knowledge domains
- Data preparation script for training
- Training script (placeholder for fine-tuning/ML models)
- FastAPI server for serving your AI over HTTP
- Dockerfile for containerized deployment

## Run Locally
```bash
pip install -r requirements.txt
python main.py
```

## Train
```bash
python data_prep.py
python train.py
```

## Serve with FastAPI
```bash
uvicorn serve:app --reload --host 0.0.0.0 --port 8000
```

## Build Docker
```bash
docker build -t cyber-ai .
docker run -p 8000:8000 cyber-ai
```