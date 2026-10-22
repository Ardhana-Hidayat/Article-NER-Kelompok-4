"""
Step 3: NER Model Training Pipeline
- Evaluates baseline model (en_core_web_sm)
- Fine-tunes spaCy transition-based NER model on training split
- Evaluates fine-tuned model on validation and held-out test splits
- Exports best model to models/ner_finetuned and metrics to data/evaluation_results.json
"""
from collections import defaultdict
from pathlib import Path
import json
import random
import time
import numpy as np
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "ner_finetuned"
LABELS = ["PERSON", "ORG", "GPE", "DATE", "EVENT"]
LABEL_MAP = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "GPE", "LOC": "GPE", "DATE": "DATE", "TIME": "DATE", "EVENT": "EVENT"}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_spans(entities: list[dict]) -> list[tuple]:
    """Sort and filter non-overlapping entities."""
    spans = [(e["start"], e["end"], e["label"]) for e in entities if e["label"] in LABELS]
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    valid = []
    for s in spans:
        if not valid or s[0] >= valid[-1][1]:
            valid.append(s)
    return valid


def evaluate(nlp, dataset: list[dict], label_map: dict = None) -> dict:
    """Exact-span matching evaluation per-class and overall."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)

    for item in dataset:
        text = item["combined_text"]
        doc = nlp(text)
        pred = {(ent.start_char, ent.end_char, label_map.get(ent.label_, ent.label_) if label_map else ent.label_)
                for ent in doc.ents if (label_map.get(ent.label_, ent.label_) if label_map else ent.label_) in LABELS}
        gold = {(e["start"], e["end"], e["label"]) for e in item["entities"] if e["label"] in LABELS}

        for p in pred:
            if p in gold:
                tp[p[2]] += 1
            else:
                fp[p[2]] += 1
        for g in gold:
            if g not in pred:
                fn[g[2]] += 1

    metrics = {}
    for l in LABELS:
        p = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
        r = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[l] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": tp[l] + fn[l]}

    tot_tp, tot_fp, tot_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    ov_p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    ov_r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    ov_f1 = 2 * ov_p * ov_r / (ov_p + ov_r) if (ov_p + ov_r) > 0 else 0.0
    metrics["OVERALL"] = {"precision": round(ov_p, 4), "recall": round(ov_r, 4), "f1": round(ov_f1, 4), "support": tot_tp + tot_fn}
    return metrics


def train_model(train_raw: list[dict], val_raw: list[dict], epochs: int = 30):
    """Train spaCy blank('en') NER with early stopping."""
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for l in LABELS:
        ner.add_label(l)

    random.seed(SEED)
    np.random.seed(SEED)
    spacy.util.fix_random_seed(SEED)
    optimizer = nlp.begin_training()

    train_data = [(d["combined_text"], {"entities": clean_spans(d["entities"])}) for d in train_raw if len(d["combined_text"]) <= 50000]
    best_f1, best_epoch, no_imp, patience = 0.0, 0, 0, 5
    start_t = time.time()

    print(f"Training on {len(train_data)} documents...")
    for ep in range(epochs):
        random.shuffle(train_data)
        losses = {}
        for batch in minibatch(train_data, size=compounding(4.0, 8.0, 1.001)):
            examples = [Example.from_dict(nlp.make_doc(t), ann) for t, ann in batch]
            nlp.update(examples, drop=0.35, losses=losses, sgd=optimizer)

        val_res = evaluate(nlp, val_raw)
        f1 = val_res["OVERALL"]["f1"]
        print(f"Epoch {ep+1:2d} | Loss: {losses.get('ner', 0):.2f} | Val F1: {f1:.4f} (P: {val_res['OVERALL']['precision']:.4f}, R: {val_res['OVERALL']['recall']:.4f})")

        if f1 > best_f1:
            best_f1, best_epoch, no_imp = f1, ep + 1, 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            nlp.to_disk(MODEL_DIR)
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    duration = round(time.time() - start_t, 1)
    print(f"Training finished in {duration}s. Best Epoch: {best_epoch} (Val F1: {best_f1:.4f})")
    return spacy.load(MODEL_DIR), {"duration": duration, "best_epoch": best_epoch, "val_f1": best_f1}


def run_pipeline():
    train_data = load_jsonl(DATA_DIR / "annotations_train.jsonl")
    val_data = load_jsonl(DATA_DIR / "annotations_val.jsonl")
    test_data = load_jsonl(DATA_DIR / "annotations_test.jsonl")

    # Baseline evaluation
    base_nlp = spacy.load("en_core_web_sm")
    base_nlp.max_length = 200000
    base_val = evaluate(base_nlp, val_data, LABEL_MAP)
    base_test = evaluate(base_nlp, test_data, LABEL_MAP)

    # Train model
    best_nlp, meta = train_model(train_data, val_data)
    trained_val = evaluate(best_nlp, val_data)
    trained_test = evaluate(best_nlp, test_data)

    # Save evaluation summary
    results = {
        "seed": SEED,
        "matching_scheme": "exact-span matching",
        "training_meta": meta,
        "baseline": {"validation": base_val, "test": base_test},
        "trained": {"validation": trained_val, "test": trained_test},
    }
    (DATA_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFinal Test Results: F1={trained_test['OVERALL']['f1']:.4f}, P={trained_test['OVERALL']['precision']:.4f}, R={trained_test['OVERALL']['recall']:.4f}")
    print(f"Evaluation metrics saved -> {DATA_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    run_pipeline()
