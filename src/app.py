"""
FastAPI Backend for Article Named-Entity Recognition Service.
Exposes POST /extract-entities and serves Web UI at GET /.
"""
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import spacy
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "ner_finetuned"
TEMPLATE_CANDIDATES = [
    Path(__file__).resolve().parent / "templates" / "index.html",
    Path(__file__).resolve().parent / "template" / "index.html",
]
TEMPLATE_PATH = next((p for p in TEMPLATE_CANDIDATES if p.exists()), TEMPLATE_CANDIDATES[0])
MAX_COMBINED_LENGTH = 50000
MODEL_NAME = "spacy-ner-finetuned-maybank-articles"
SUPPORTED_LABELS = {"PERSON", "ORG", "GPE", "DATE", "EVENT"}

# Load model (fallback to baseline if fine-tuned model directory not ready)
try:
    nlp = spacy.load(str(MODEL_DIR))
except Exception:
    nlp = spacy.load("en_core_web_sm")
nlp.max_length = MAX_COMBINED_LENGTH + 1000

app = FastAPI(title="Article NER Service", version="1.0.0")


def validate_field(data: dict, field: str) -> str:
    """Validate presence, type, and non-empty content of string field."""
    if field not in data or data[field] is None:
        raise HTTPException(400, detail={"error": "Missing field", "message": f"'{field}' field is required."})
    val = data[field]
    if not isinstance(val, str):
        raise HTTPException(400, detail={"error": "Invalid type", "message": f"'{field}' must be a string."})
    if not val.strip():
        raise HTTPException(400, detail={"error": "Empty field", "message": f"'{field}' must not be empty or whitespace-only."})
    return val.strip()


@app.post("/extract-entities")
async def extract_entities(request: Request):
    """Extract named entities from article title and body."""
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError()
    except Exception:
        raise HTTPException(400, detail={"error": "Invalid JSON", "message": "Request body must be valid JSON object."})

    title = validate_field(body, "title")
    body_text = validate_field(body, "body")
    combined = f"{title}\n\n{body_text}"

    if len(combined) > MAX_COMBINED_LENGTH:
        raise HTTPException(400, detail={"error": "Text too long", "message": f"Combined length ({len(combined)}) exceeds maximum of {MAX_COMBINED_LENGTH}."})

    doc = nlp(combined)
    entities = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents if ent.label_ in SUPPORTED_LABELS
    ]

    return {"entities": entities, "meta": {"entity_count": len(entities), "model": MODEL_NAME}}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the Web UI."""
    if TEMPLATE_PATH.exists():
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Article NER Service</h1><p>UI template not found.</p>")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)