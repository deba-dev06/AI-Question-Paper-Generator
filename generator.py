from transformers import pipeline
import time

qg_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

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
    results = qg_pipeline(
        prompt,
        max_length=256,
        num_return_sequences=5,
        do_sample=True,
        temperature=0.8
    )
    _t_elapsed = round(time.perf_counter() - _t_start, 2)

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