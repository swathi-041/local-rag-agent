import json
from PyPDF2 import PdfReader

def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join([p.extract_text() for p in reader.pages])

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2)
