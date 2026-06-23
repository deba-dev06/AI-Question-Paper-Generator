import pdfplumber

def read_file(file):
    if hasattr(file, "read"):
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    else:
        with open(file.name, "r", encoding="utf-8") as f:
            return f.read()