from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

_MODEL_NAME = "google/flan-t5-base"
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = _model.to(_device)

def truncate_text(text, max_words=300):
    words = text.split()
    return " ".join(words[:max_words])

def generate_questions(text, marks):
    text = truncate_text(text)  # ← moved here, before prompt is built
    
    if marks == 2:
        instruction = "Generate 5 very short questions (1 line each)."
    elif marks == 3:
        instruction = "Generate 5 medium-length questions."
    elif marks == 5:
        instruction = "Generate 5 descriptive questions."
    else:
        instruction = "Generate 5 questions."

    prompt = f"{instruction}\nText:\n{text}"
    _t_start = time.perf_counter()
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(_device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=5,
            do_sample=True,
            temperature=0.8,
        )
    _t_elapsed = round(time.perf_counter() - _t_start, 2)
    results = [
        {"generated_text": _tokenizer.decode(ids, skip_special_tokens=True)}
        for ids in output_ids
    ]

    questions = []
    for r in results:
        lines = r["generated_text"].split("\n")
        for q in lines:
            q = q.strip()
            if q:
                questions.append(q)

    unique_questions = list(dict.fromkeys(questions))
    final_questions = unique_questions[:10]

    return "\n".join(final_questions), _t_elapsed