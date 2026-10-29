from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SUPPORTED_LANGUAGES = {
        "eng_Latn": "English",
        "nya_Latn": "Chichewa",
        "fra_Latn": "French",
        "zho_Hans": "Chinese (Simplified)",
        "zho_Hant": "Chinese (Traditional)",
        "hin_Deva": "Hindi",
        "ben_Beng": "Bengali",
        "tam_Taml": "Tamil",
        "tel_Telu": "Telugu",
        "mar_Deva": "Marathi",
        "guj_Gujr": "Gujarati",
        "pan_Guru": "Punjabi",
        "urd_Arab": "Urdu",
    }

app = FastAPI(title="Lankhula Translation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    source_lang: str = "eng_Latn"
    target_lang: str = "nya_Latn"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate")
def translate(request: TranslationRequest):
    text = request.text.strip()
    source_lang = request.source_lang
    target_lang = request.target_lang

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text is required",
        )

    if source_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported source language",
        )

    if target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported target language",
        )

    if source_lang == target_lang:
        raise HTTPException(
            status_code=400,
            detail="Source and target languages must be different",
        )

    try:
        tokenizer, model = load_model()

        # Tell NLLB which language the input text uses.
        tokenizer.src_lang = source_lang

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        target_token_id = tokenizer.convert_tokens_to_ids(
            target_lang
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_new_tokens=256,
                num_beams=3,
            )

        translated = tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
        )[0]

        return {
            "source": text,
            "translation": translated,
            "source_language": source_lang,
            "target_language": target_lang,
        }

    except Exception as exc:
        # Log the actual error in your backend terminal.
        print(f"Translation error: {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Translation failed. Check the backend logs.",
        ) from exc