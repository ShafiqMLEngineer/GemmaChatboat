# 🤖 AI Mock Interview Chatbot (Gemma 3 1B + QLoRA)

A conversational AI chatbot fine-tuned on a custom interview Q&A dataset using **Google Gemma-3-1B-Instruct** with **QLoRA** and the Hugging Face ecosystem.

## 🚀 Features

- Fine-tuned Gemma-3-1B-Instruct
- QLoRA (4-bit Quantization)
- Custom Chat Dataset
- Hugging Face Transformers
- Dynamic Padding
- Validation & Perplexity Evaluation
- Hugging Face Hub Deployment
- and GUI Deploy on Streamlit Cloud
---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- BitsAndBytes
- Datasets
- Trainer API

---

## 📊 Training Results

| Metric | Value |
|--------|------:|
| Training Loss | 0.3753 |
| Validation Loss | 0.4680 |
| Perplexity | 1.60 |

---

## 🔄 Project Workflow

1. Prepare Chat Dataset
2. Apply Chat Template
3. Tokenize Dataset
4. Train / Validation Split
5. Dynamic Padding
6. Load Gemma-3-1B-Instruct
7. Apply QLoRA
8. Fine-tune Model
9. Evaluate Model
10. Calculate Perplexity
11. Push Model & Tokenizer to Hugging Face
12. Perform Chat Inference

---

## 💬 Example

**User**

```
Explain briefly: What is the AI Mock Interview project?
```

**Assistant**

```
The AI Mock Interview project is a full-stack AI application that generates personalized interview questions and provides performance feedback based on the user's profile.
```

---

## 🤗 Hugging Face

- Model: `shafiq433/GemmaChatBoat`

---

## 📌 Future Improvements

- Larger Dataset
- Better Response Quality
- RAG Integration
- FastAPI Deployment
- Streamlit Web Interface

---

⭐ If you found this project helpful, consider giving it a star!
