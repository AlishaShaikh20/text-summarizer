import modal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


MODEL_NAME = "AlishaShaikh20/text-summarizer-t5"


def download_model():
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    T5Tokenizer.from_pretrained(MODEL_NAME)
    T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sentencepiece",
        "fastapi[standard]",
    )
    .run_function(download_model)
    .add_local_file("index.html", "/root/index.html")
)


app = modal.App("text-summariser", image=image)


@app.cls()
class SummarizerModel:

    @modal.enter()
    def load_model(self):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

        self.model.to("cpu")
        self.model.eval()

    @modal.method()
    def summarize(self, dialogue: str) -> str:
        import re
        import torch

        dialogue = re.sub(r"\r\n", " ", dialogue)
        dialogue = re.sub(r"\s+", " ", dialogue)
        dialogue = re.sub(r"<.*?>", " ", dialogue)
        dialogue = dialogue.strip().lower()

        inputs = self.tokenizer(
            dialogue,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to("cpu")
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_beams=4,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return summary


web_app = FastAPI()


web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
    text: str


@web_app.get("/", response_class=HTMLResponse)
def home():
    with open("/root/index.html", "r", encoding="utf-8") as file:
        return file.read()


@web_app.post("/summarize")
def api_summarize(request: SummarizeRequest):
    model = SummarizerModel()

    summary = model.summarize.remote(request.text)

    return {"summary": summary}


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app