# monitor/inference_local.py
# Enhanced local inference with CONTEXT-AWARE cautionary words logic

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
from typing import List, Dict, Tuple

DEFAULT_MODEL_DIR = os.getenv("INFERENCE_MODEL_DIR", "/data/model")
DEFAULT_MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

# UPDATED: More focused caution keywords (removed overly generic ones)
CAUTION_KEYWORDS = [
    # Strong negative sentiment
    "i am unhappy", "i'm unhappy", "not happy", "not satisfied", "dissatisfied",
    "disappointed", "unacceptable", "frustrated", "angry", "bad experience",
    "terrible", "awful", "poor quality",
    
    # Unresolved issues with escalation
    "unresolved", "still not fixed", "issue persists", "still waiting",
    "weeks now", "no response", "not working", "does not work",
    "recurring issue", "repeat issue", "keeps happening",
    
    # Escalation language
    "escalate", "escalation", "higher management", "need immediate attention", 
    "urgent action", "immediate action", "requires urgent", "complaint",
    
    # Second thoughts / doubts
    "second thoughts", "reconsidering", "questioning whether", 
    "having doubts", "not confident", "losing confidence",
    
    # Problems without solutions
    "problem with", "issue with", "trouble with", "unable to resolve",
    "gaps in", "missing information", "incomplete", "lack of",
    "below expectations", "not up to the mark", "not acceptable",
    
    # Frustration indicators
    "very frustrated", "extremely disappointed", "not pleased",
    "this is unacceptable", "needs to be fixed immediately"
]

# Context indicators that suggest NEUTRAL even with keywords
NEUTRAL_CONTEXT_INDICATORS = [
    # Explanatory/informative
    "we were fixing", "we have fixed", "we resolved", "now completed",
    "has been resolved", "issue is fixed", "working on",
    
    # Polite professional
    "please find attached", "kindly review", "for your reference",
    "let me know if", "thank you", "thanks", "regards",
    
    # Auto-reply templates
    "out of office", "currently unavailable", "will respond upon my return",
    "limited access", "away from office",
    
    # Status updates (not complaints)
    "fyi", "for your information", "updating you", "status update",
    "progress update", "just to inform"
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

def has_neutral_context(text: str) -> bool:
    """
    Check if text has neutral context indicators.
    If these are present, likely not a complaint even with caution keywords.
    """
    if not text:
        return False
    
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in NEUTRAL_CONTEXT_INDICATORS)

def contains_caution(text: str, keywords: List[str] = None) -> Tuple[bool, str]:
    """
    UPDATED: Context-aware caution keyword detection.
    Returns (has_keyword, matched_keyword)
    
    Now checks for neutral context before flagging as cautionary.
    """
    if keywords is None:
        keywords = CAUTION_KEYWORDS
    
    if not text or not keywords:
        return False, ""
    
    text_lower = text.lower()
    
    # Check for neutral context first
    has_neutral = has_neutral_context(text)
    
    # Check each keyword
    for kw in keywords:
        if not kw:
            continue
        kw_lower = str(kw).strip().lower()
        if kw_lower and kw_lower in text_lower:
            # If neutral context present, don't flag generic keywords
            if has_neutral and kw_lower in [
                "concern", "urgent", "please update", "delay", "issue",
                "follow up", "following up", "any update"
            ]:
                continue  # Skip - neutral context overrides
            
            # Otherwise, flag as cautionary
            return True, kw
    
    return False, ""

def apply_caution_postprocess(pred_idx: int, 
                              probs: np.ndarray, 
                              text: str,
                              keywords: List[str] = None,
                              neg_prob_thresh: float = 0.10) -> Tuple[int, str]:
    """
    UPDATED: More intelligent cautionary keyword override logic.
    
    New rules:
    1. If model predicts Negative, trust it (no override needed)
    2. If model predicts Positive, be VERY careful - only override if strong evidence
    3. If model predicts Neutral:
       - Check for neutral context first
       - If neutral context exists, keep as Neutral
       - If caution keyword + sufficient neg probability -> flag as Negative
    """
    if keywords is None:
        keywords = CAUTION_KEYWORDS
    
    p_neg = float(probs[0])
    p_neu = float(probs[1])
    p_pos = float(probs[2])
    
    # If already predicted Negative, trust the model
    if pred_idx == 0:
        return pred_idx, "original"
    
    # Check for caution keywords
    has_kw, matched_kw = contains_caution(text, keywords)
    
    if not has_kw:
        return pred_idx, "original"
    
    # Has caution keyword - now decide based on context and probabilities
    
    # If Positive prediction - be VERY conservative
    if pred_idx == 2:
        # Only override if VERY HIGH negative probability AND strong negative keywords
        strong_negative_kws = [
            "unacceptable", "disappointed", "frustrated", "angry", 
            "second thoughts", "reconsidering", "escalate", "complaint"
        ]
        has_strong_kw = any(kw in text.lower() for kw in strong_negative_kws)
        
        if has_strong_kw and p_neg >= 0.25:  # High threshold for Positive override
            return 0, f"caution_keyword:{matched_kw}(p_neg={p_neg:.3f},strong_override)"
        
        return pred_idx, "original"  # Keep as Positive
    
    # If Neutral prediction - apply intelligent override
    if pred_idx == 1:
        # Check neutral context
        if has_neutral_context(text):
            # Neutral context present - keep as Neutral unless VERY negative
            if p_neg >= 0.30:  # Very high threshold
                return 0, f"caution_keyword:{matched_kw}(p_neg={p_neg:.3f},override_despite_context)"
            return pred_idx, "neutral_context_preserved"
        
        # No neutral context - apply override if neg probability sufficient
        if p_neg >= neg_prob_thresh:
            return 0, f"caution_keyword:{matched_kw}(p_neg={p_neg:.3f})"
        
        # Neg probability too low - keep as Neutral
        return pred_idx, f"neutral_kept(p_neg={p_neg:.3f}<{neg_prob_thresh})"
    
    return pred_idx, "original"

def classify_texts(texts: List[str],
                   model_dir: str = DEFAULT_MODEL_DIR,
                   max_length: int = DEFAULT_MAX_LENGTH,
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   apply_rule: bool = True,
                   neg_prob_thresh: float = 0.10,
                   caution_keywords: List[str] = None
                   ) -> List[Dict]:
    """
    Classify multiple texts with UPDATED context-aware cautionary keyword override.
    
    Args:
        texts: List of email texts to classify
        model_dir: Path to model directory
        max_length: Max token length
        batch_size: Batch size for inference
        apply_rule: Whether to apply caution keyword override
        neg_prob_thresh: Negative probability threshold (now 0.10 default - higher)
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
                   neg_prob_thresh: float = 0.10, 
                   caution_keywords: List[str] = None) -> Dict:
    """
    Classify a single email text with context-aware logic.
    
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
    print("Testing Context-Aware Classification")
    print("="*60)
    
    # Test samples covering different scenarios
    samples = [
        # Should be NEUTRAL (polite professional)
        "Hi team, please find attached the report for your reference. Thanks!",
        
        # Should be NEUTRAL (explanatory)
        "Sorry for the delay in submission, we were fixing some issues. All reconciliation entries are now posted.",
        
        # Should be NEUTRAL (auto-reply with "urgent")
        "I am out of office with limited access. If your concern is urgent, please contact the team.",
        
        # Should be NEGATIVE (real concern needing attention)
        "I'm disappointed with the delay. This issue needs immediate attention and resolution.",
        
        # Should be NEGATIVE (second thoughts)
        "We're having second thoughts about proceeding with this engagement given the ongoing delays.",
        
        # Should be NEGATIVE (unresolved frustration)
        "This has been unresolved for weeks now. Very frustrated with the lack of progress.",
        
        # Should be POSITIVE
        "Thank you for the excellent work! The audit report looks great. Well done!"
    ]
    
    expected = [1, 1, 1, 0, 0, 0, 2]  # Expected labels
    
    print("\n--- Testing with context-aware logic ---")
    results = classify_texts(samples, apply_rule=True, neg_prob_thresh=0.10)
    
    for i, (text, result, exp) in enumerate(zip(samples, results, expected), 1):
        correct = "✅" if result['final_idx'] == exp else "❌"
        print(f"\n{correct} Sample {i}:")
        print(f"Text: {text[:100]}")
        print(f"Model: {result['pred_label']} → Final: {result['final_label']}")
        print(f"Probs: Neg={result['probs'][0]:.3f}, Neu={result['probs'][1]:.3f}, Pos={result['probs'][2]:.3f}")
        print(f"Reason: {result['postprocess_reason']}")
        print("-" * 60)