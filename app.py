import modal

# Create Modal app
app = modal.App("text-summariser")

# Build the environment for the Modal container
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sentencepiece",
    )
)


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=300,
)
class SummarizerModel:

    @modal.enter()
    def load_model(self):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        model_name = "AlishaShaikh20/text-summarizer-t5"

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.model.to("cuda")
        self.model.eval()

    @modal.method()
    def summarize(self, dialogue: str) -> str:
        import re
        import torch

        # Same preprocessing as your original app
        dialogue = re.sub(r"\r\n", " ", dialogue)
        dialogue = re.sub(r"\s+", " ", dialogue)
        dialogue = re.sub(r"<.*?>", " ", dialogue)
        dialogue = dialogue.strip().lower()

        # Tokenize
        inputs = self.tokenizer(
            dialogue,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_beams=4,
                early_stopping=True,
            )

        # Decode
        summary = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return summary