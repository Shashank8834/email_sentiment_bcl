# monitor/inference_local.py
# Local HF transformers inference. Configurable via INFERENCE_MODEL_DIR env.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
from typing import List, Dict, Tuple

DEFAULT_MODEL_DIR = os.getenv("INFERENCE_MODEL_DIR", "/data/model")
DEFAULT_MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

CAUTION_KEYWORDS = [
    "concern", "concerned", "not entirely", "not satisfied", "slower than expected",
    "unable to", "could you please", "please update", "i am unhappy", "i'm unhappy",
    "i was unable", "unresolved", "disappointed", "need an update", "not happy", "worry"
]

ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

_tokenizer = None
_model = None
_device = None

def _load_model_and_tokenizer(model_dir: str = DEFAULT_MODEL_DIR):
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model, _device

    # Accept local folder or HF id
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    try:
        model.config.id2label = {k: v for k, v in ID2LABEL.items()}
        model.config.label2id = {v: k for k, v in ID2LABEL.items()}
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _tokenizer, _model, _device = tokenizer, model, device
    return _tokenizer, _model, _device

def _softmax(logits: np.ndarray) -> np.ndarray:
    ex = np.exp(logits - np.max(logits))
    return ex / ex.sum(axis=-1, keepdims=True)

def contains_caution(text: str, keywords: List[str] = CAUTION_KEYWORDS) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in keywords)

def apply_caution_postprocess(pred_idx: int, probs: np.ndarray, text: str,
                              keywords: List[str] = None,
                              neg_prob_thresh: float = 0.06) -> Tuple[int, str]:
    if keywords is None:
        keywords = CAUTION_KEYWORDS
    p_neg = float(probs[0])
    has_kw = contains_caution(text, keywords)
    if (pred_idx in [1, 2]) and has_kw and (p_neg >= neg_prob_thresh):
        return 0, f"forced_negative_by_rule(p_neg={p_neg:.3f})"
    return pred_idx, "original"

def classify_texts(texts: List[str],
                   model_dir: str = DEFAULT_MODEL_DIR,
                   max_length: int = DEFAULT_MAX_LENGTH,
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   apply_rule: bool = True,
                   neg_prob_thresh: float = 0.06,
                   caution_keywords: List[str] = None
                   ) -> List[Dict]:
    if caution_keywords is None:
        caution_keywords = CAUTION_KEYWORDS

    tokenizer, model, device = _load_model_and_tokenizer(model_dir)
    results = []
    texts = [("" if t is None else str(t)) for t in texts]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.cpu().numpy()
        probs_batch = np.array([_softmax(l) for l in logits])
        preds_batch = np.argmax(probs_batch, axis=1)

        for txt, pred_idx, probs in zip(batch, preds_batch, probs_batch):
            pred_idx = int(pred_idx)
            pred_label = ID2LABEL.get(pred_idx, str(pred_idx))
            final_idx, reason = (pred_idx, "original")
            if apply_rule:
                final_idx, reason = apply_caution_postprocess(pred_idx, probs, txt, caution_keywords, neg_prob_thresh)
            final_label = ID2LABEL.get(final_idx, str(final_idx))
            results.append({
                "pred_idx": pred_idx,
                "pred_label": pred_label,
                "probs": [float(probs[0]), float(probs[1]), float(probs[2])],
                "final_idx": int(final_idx),
                "final_label": final_label,
                "postprocess_reason": reason
            })
    return results

def classify_email(text: str, model_dir: str = DEFAULT_MODEL_DIR, max_length: int = DEFAULT_MAX_LENGTH, apply_rule: bool = True, neg_prob_thresh: float = 0.06, caution_keywords: List[str] = None) -> Dict:
    res = classify_texts([text], model_dir=model_dir, max_length=max_length, batch_size=1, apply_rule=apply_rule, neg_prob_thresh=neg_prob_thresh, caution_keywords=caution_keywords)
    return res[0] if res else {"pred_idx": None, "pred_label": None, "probs": [0.0,0.0,0.0], "final_idx": None, "final_label": None, "postprocess_reason": "error"}

if __name__ == "__main__":
    samples = [
        "Hi team, thank you — the audit report looks great. Well done!",
        "I am disappointed. The issue reported earlier is unresolved and unacceptable.",
        "Can you please update the status? I am a bit concerned about the timeline."
    ]
    out = classify_texts(samples)
    for t, r in zip(samples, out):
        print("TEXT:", t[:120])
        print("->", r)
        print("-" * 40)
