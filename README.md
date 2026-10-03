# Text Summarizer — T5 + FastAPI

A text summarization project using a fine-tuned **T5-small** transformer model, served through a FastAPI web application with a simple browser-based interface.

The project primarily focuses on summarizing conversational text.

## Live Demo

**[Try the Text Summarizer](https://alishashaikh20--text-summariser-fastapi-app.modal.run/)**

## Overview

This project explores fine-tuning a pretrained **T5 (Text-to-Text Transfer Transformer)** model for text summarization.

The fine-tuned model is connected to a FastAPI backend for inference, with a simple HTML/CSS/JavaScript frontend where users can enter text and receive a generated summary.

The application is deployed using **Modal**.

## Model

- **Base model:** `t5-small`
- **Fine-tuning:** SAMSum dataset
- **Training samples:** 4,000
- **Validation samples:** 500
- **Epochs:** 3
- **Batch size:** 3
- **Weight decay:** 0.01
- **Warmup steps:** 400
- **Final training loss:** ~0.762
- **Training hardware:** NVIDIA Tesla T4 GPU

**Fine-tuned Model:**  
[AlishaShaikh20/text-summarizer-t5](https://huggingface.co/AlishaShaikh20/text-summarizer-t5)

## Workflow

```text
SAMSum Dataset
      ↓
Data Cleaning
      ↓
T5 Tokenization
      ↓
T5-small Fine-tuning
      ↓
Fine-tuned Model
      ↓
FastAPI Backend
      ↓
HTML / CSS / JavaScript Frontend
      ↓
Generated Summary
```

## Example

**Input:**

> Sarah: Hey, are we still meeting tomorrow for the project?  
> John: Yes, I think 10 AM works.  
> Sarah: Perfect. Should we meet at the library?  
> John: The library is fine. I'll bring the project notes.  
> Sarah: Great. I'll bring my laptop and the presentation slides.  
> John: Sounds good. See you tomorrow at 10.

**Generated Summary:**

> Sarah and John are meeting tomorrow at 10 AM for the project. John will bring the project notes.

*Generated summaries may vary depending on the input.*

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- T5-small
- FastAPI
- Jinja2
- HTML / CSS / JavaScript
- Pandas
- Modal

## Project Structure

```text
text_summariser/
│
├── app.py                  # FastAPI application
├── modal_app.py            # Modal deployment configuration
├── index.html              # Frontend interface
├── text-summarizer.ipynb   # Model training and experimentation
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

The fine-tuned model weights and dataset files are not included in the repository.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AlishaShaikh20/text-summarizer.git
cd text-summarizer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI application

```bash
uvicorn app:app --reload
```

### 5. Open in your browser

```text
http://127.0.0.1:8000
```

## What I Learned

- Fine-tuning a pretrained Transformer model
- T5 encoder-decoder architecture
- Hugging Face tokenization
- Sequence-to-sequence text generation
- Hugging Face `Trainer`
- Beam search decoding
- Building an inference API with FastAPI
- Connecting an ML model to a frontend
- Deploying an ML application using Modal

## Limitations

This project was trained on a limited subset of the SAMSum dataset, so summary quality can vary depending on the input.

The model is primarily suited to conversational text similar to the data used during fine-tuning and should not be considered a general-purpose document summarization
