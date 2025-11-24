# monitor/inference_local.py
# Local HF transformers inference with FIXED cautionary words logic

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
from typing import List, Dict, Tuple

DEFAULT_MODEL_DIR = os.getenv("INFERENCE_MODEL_DIR", "/data/model")
DEFAULT_MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

# Default caution keywords (fallback if DB is empty)
CAUTION_KEYWORDS = [
    "concern", "concerned", "not entirely", "not satisfied", "slower than expected",
    "unable to", "could you please", "please update", "i am unhappy", "i'm unhappy",
    "i was unable", "unresolved", "disappointed", "need an update", "not happy", 
    "worry", "there are gaps", "missing information", "bad experience", "angry", 
    "dissatisfied", "frustrated", "issue", "resolve the problem", "urgent", 
    "complaint", "fix the gaps",
    "any update", "waiting for your response", "still waiting", "follow up", 
    "following up", "kindly update", "kindly revert", "please revert", 
    "waiting since", "any progress", "status update", "need clarity", 
    "seeking clarification", "pending from your side", "not acceptable", 
    "below expectations", "not up to the mark", "not working as expected", 
    "did not receive", "delayed response", "delay in", "taking too long", 
    "too slow", "still unresolved", "still not fixed", "issue persists", 
    "recurring issue", "repeat issue", "keeps happening", 
    "need immediate attention", "urgent action", "needs to be resolved", 
    "escalate", "escalation", "higher management", "please take action", 
    "requires attention", "requires correction", "please look into this", 
    "incomplete", "incorrect information", "incorrect details", "not working", 
    "does not work", "problem with", "trouble with", "facing trouble", "lack of"
]


ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

_tokenizer = None
_model = None
_device = None

def _load_model_and_tokenizer(model_dir: str = DEFAULT_MODEL_DIR):
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model, _device

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

def contains_caution(text: str, keywords: List[str] = None) -> Tuple[bool, str]:
    """
    Check if text contains any caution keywords.
    Returns (has_keyword, matched_keyword)
    
    FIXED: Normalizes both text and keywords to lowercase for matching
    """
    if keywords is None:
        keywords = CAUTION_KEYWORDS
    
    if not text or not keywords:
        return False, ""
    
    # Normalize text to lowercase
    text_lower = text.lower()
    
    # Check each keyword (normalize to lowercase)
    for kw in keywords:
        if not kw:
            continue
        kw_lower = str(kw).strip().lower()
        if kw_lower and kw_lower in text_lower:
            return True, kw
    
    return False, ""

def apply_caution_postprocess(pred_idx: int, 
                              probs: np.ndarray, 
                              text: str,
                              keywords: List[str] = None,
                              neg_prob_thresh: float = 0.06) -> Tuple[int, str]:
    """
    Apply cautionary keyword override logic.
    
    FIXED LOGIC:
    - If caution keyword is found AND prediction is NOT negative -> force to Negative
    - The neg_prob_thresh is now optional - keyword presence is primary trigger
    """
    if keywords is None:
        keywords = CAUTION_KEYWORDS
    
    p_neg = float(probs[0])
    has_kw, matched_kw = contains_caution(text, keywords)
    
    # NEW LOGIC: If caution keyword found and prediction is Neutral or Positive
    if has_kw and pred_idx in [1, 2]:
        # Force to negative regardless of threshold
        # (threshold is just for logging/debugging now)
        return 0, f"caution_keyword:{matched_kw}(p_neg={p_neg:.3f})"
    
    # Alternative stricter logic (uncomment if you want threshold requirement):
    # if has_kw and pred_idx in [1, 2] and p_neg >= neg_prob_thresh:
    #     return 0, f"caution_keyword:{matched_kw}(p_neg={p_neg:.3f})"
    
    return pred_idx, "original"

def classify_texts(texts: List[str],
                   model_dir: str = DEFAULT_MODEL_DIR,
                   max_length: int = DEFAULT_MAX_LENGTH,
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   apply_rule: bool = True,
                   neg_prob_thresh: float = 0.06,
                   caution_keywords: List[str] = None
                   ) -> List[Dict]:
    """
    Classify multiple texts with optional cautionary keyword override.
    
    Args:
        texts: List of email texts to classify
        model_dir: Path to model directory
        max_length: Max token length
        batch_size: Batch size for inference
        apply_rule: Whether to apply caution keyword override
        neg_prob_thresh: Negative probability threshold (now optional)
        caution_keywords: List of keywords to check (uses default if None)
    
    Returns:
        List of classification results with override information
    """
    if caution_keywords is None:
        caution_keywords = CAUTION_KEYWORDS
    
    # Normalize keywords to lowercase
    caution_keywords = [str(kw).strip().lower() for kw in caution_keywords if kw and str(kw).strip()]

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
            
            # Apply caution keyword override if enabled
            final_idx, reason = (pred_idx, "original")
            if apply_rule:
                final_idx, reason = apply_caution_postprocess(
                    pred_idx, probs, txt, caution_keywords, neg_prob_thresh
                )
            
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

def classify_email(text: str, 
                   model_dir: str = DEFAULT_MODEL_DIR, 
                   max_length: int = DEFAULT_MAX_LENGTH, 
                   apply_rule: bool = True, 
                   neg_prob_thresh: float = 0.06, 
                   caution_keywords: List[str] = None) -> Dict:
    """
    Classify a single email text.
    
    Returns classification result with cautionary keyword override if applicable.
    """
    res = classify_texts(
        [text], 
        model_dir=model_dir, 
        max_length=max_length, 
        batch_size=1, 
        apply_rule=apply_rule, 
        neg_prob_thresh=neg_prob_thresh, 
        caution_keywords=caution_keywords
    )
    return res[0] if res else {
        "pred_idx": None, 
        "pred_label": None, 
        "probs": [0.0, 0.0, 0.0], 
        "final_idx": None, 
        "final_label": None, 
        "postprocess_reason": "error"
    }

# Test script
if __name__ == "__main__":
    print("="*60)
    print("Testing Cautionary Keywords Implementation")
    print("="*60)
    
    # Test samples
    samples = [
        "Hi team, thank you — the audit report looks great. Well done!",
        "I am disappointed. The issue reported earlier is unresolved and unacceptable.",
        "Can you please update the status? I am a bit concerned about the timeline.",
        "There are gaps in the documentation that need to be addressed.",
        "Everything looks good, no issues to report."
    ]
    
    # Test with default keywords
    print("\n--- Testing with default keywords ---")
    out = classify_texts(samples, apply_rule=True)
    for t, r in zip(samples, out):
        print(f"\nTEXT: {t[:100]}")
        print(f"Prediction: {r['pred_label']} (probs: {[f'{p:.3f}' for p in r['probs']]})")
        print(f"Final: {r['final_label']} - {r['postprocess_reason']}")
        print("-" * 60)
    
    # Test with custom keywords
    print("\n--- Testing with custom keywords ---")
    custom_keywords = ["gaps", "missing", "incomplete"]
    out2 = classify_texts(samples, apply_rule=True, caution_keywords=custom_keywords)
    for t, r in zip(samples, out2):
        print(f"\nTEXT: {t[:100]}")
        print(f"Prediction: {r['pred_label']} (probs: {[f'{p:.3f}' for p in r['probs']]})")
        print(f"Final: {r['final_label']} - {r['postprocess_reason']}")
        print("-" * 60)