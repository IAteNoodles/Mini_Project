"""Utilities for running bias classification with LIME and SHAP explanations."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Dict, List, Sequence, cast

import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_text import LimeTextExplainer
from shap import Explanation
from shap.maskers import Text
from shap.models import TransformersPipeline
from transformers import AutoTokenizer, pipeline

MODEL_NAME = "himel7/bias-detector"
LABEL_DISPLAY_MAP = {
    "LABEL_0": "Non-biased",
    "LABEL_1": "Biased",
}

_NUMPY_FLOAT_PATTERN = re.compile(r"np\.float(?:16|32|64)?\(([^)]+)\)")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


import logging

logger = logging.getLogger(__name__)

@dataclass
class BiasPrediction:
    """Container for a single classifier prediction."""

    label: str
    score: float


class BiasDetector:
    """Wraps a Hugging Face pipeline and produces LIME + SHAP explanations."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        device_index = 0 if torch.cuda.is_available() else -1
        self._hf_pipeline = pipeline(
            task="text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device_index,
            top_k=None,
            return_all_scores=True,
            batch_size=32,
            model_kwargs={"attn_implementation": "eager"},
        )
        self._model = self._hf_pipeline.model
        self._tokenizer = self._hf_pipeline.tokenizer
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = getattr(self._model, "config", None)
        if config and getattr(config, "id2label", None):
            self.class_names = [config.id2label[idx] for idx in range(config.num_labels)]
        else:
            num_labels = getattr(config, "num_labels", 2) if config else 2
            self.class_names = [f"LABEL_{idx}" for idx in range(num_labels)]
        self.display_map = {label: LABEL_DISPLAY_MAP.get(label, label) for label in self.class_names}
        self.display_names = [self.display_map[label] for label in self.class_names]
        self._explainer = LimeTextExplainer(class_names=self.class_names)
        text_masker = Text(tokenizer=self._tokenizer)
        self._shap_model = TransformersPipeline(
            self._hf_pipeline,
            rescale_to_logits=True,
        )
        self._shap_explainer = shap.Explainer(
            self._shap_model,
            text_masker,
            output_names=self.display_names,
        )

    def truncate(self, text: str, max_length: int = 500) -> str:
        """Truncate text to the model's maximum sequence length."""
        tokens = self._tokenizer.encode(text, truncation=True, max_length=max_length, add_special_tokens=False)
        return self._tokenizer.decode(tokens)

    def predict(self, text: str) -> BiasPrediction:
        """Run the classifier on a single text sample."""
        outputs = self._hf_pipeline(text, truncation=True)
        scores = outputs[0] if isinstance(outputs, list) else outputs
        score_seq = cast(Sequence[Dict[str, Any]], scores)
        best = max(score_seq, key=lambda entry: float(entry["score"]))
        return BiasPrediction(label=str(best["label"]), score=float(best["score"]))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return class probability vectors for LIME."""
        outputs = self._hf_pipeline(list(texts), truncation=True)
        probabilities: List[List[float]] = []
        for item in outputs:
            entry_list = cast(Sequence[Dict[str, Any]], item)
            label_scores = {str(entry["label"]): float(entry["score"]) for entry in entry_list}
            probabilities.append([label_scores[label] for label in self.class_names])
        return np.array(probabilities)

    def explain_lime(self, text: str, num_features: int = 10, num_samples: int = 1000):
        """Generate a LIME explanation for the provided text."""
        return self._explainer.explain_instance(
            text_instance=text,
            classifier_fn=self.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
        )

    def explain(self, text: str, num_features: int = 10):
        """Backward-compatible alias that returns a LIME explanation."""
        return self.explain_lime(text, num_features=num_features)

    def shap_explain(self, text: str, max_evals: int | None = None) -> Explanation:
        """Return a SHAP Explanation object for the provided text."""
        kwargs: Dict[str, Any] = {}
        if max_evals is not None:
            kwargs["max_evals"] = max_evals
        return self._shap_explainer([text], **kwargs)

    def shap_token_matrix(
        self,
        explanation: Explanation,
        sample_index: int = 0,
    ) -> pd.DataFrame:
        """Return SHAP contributions for all labels per token."""
        sample_exp = explanation[sample_index]
        tokens = np.array(sample_exp.data, dtype=object)
        values = np.asarray(sample_exp.values)

        if values.ndim == 1:
            values = values[:, np.newaxis]
        elif values.ndim == 3:
            values = values[0]
        elif values.shape[0] != len(tokens):
            values = values.T

        df = pd.DataFrame({"Token": tokens})
        for idx, label in enumerate(self.display_names[: values.shape[1]]):
            df[label] = values[:, idx]
        df = df[df["Token"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
        return df

    def shap_dataframe(
        self,
        explanation: Explanation,
        target_label: str,
        sample_index: int = 0,
    ) -> pd.DataFrame:
        """Convert a SHAP explanation to a token-level dataframe for one label."""
        matrix = self.shap_token_matrix(explanation, sample_index)
        label_key = self.display_label(target_label)
        if label_key not in matrix.columns:
            raise ValueError(f"Label {label_key} not present in SHAP explanation")
        df = matrix[["Token", label_key]].rename(columns={label_key: "SHAP Value"})
        return df

    def shap_text_html(
        self,
        explanation: Explanation,
        target_label: str,
        sample_index: int = 0,
    ) -> str:
        """Return the HTML snippet for a SHAP text heatmap plot for a label."""
        label_key = self.display_label(target_label)
        label_slice = explanation[sample_index][:, label_key]
        shap_html = shap.plots.text(label_slice, display=False)
        raw_html = getattr(shap_html, "data", str(shap_html))
        return _NUMPY_FLOAT_PATTERN.sub(r"\1", raw_html)

    def shap_label_slice(
        self,
        explanation: Explanation,
        target_label: str,
        sample_index: int = 0,
    ) -> Explanation:
        """Return a SHAP Explanation slice for a label with readable tokens."""
        label_key = self.display_label(target_label)
        label_slice = explanation[sample_index][:, label_key]
        tokens = np.asarray(label_slice.data, dtype=object)
        cleaned_tokens = np.array([tok if str(tok).strip() else "[space]" for tok in tokens], dtype=object)
        values = np.expand_dims(np.asarray(label_slice.values), axis=0)
        data = np.expand_dims(cleaned_tokens, axis=0)
        base = np.asarray(label_slice.base_values)
        if base.shape == ():
            base = np.array([base])
        feature_names = cleaned_tokens.tolist()
        return Explanation(
            values=values,
            base_values=base,
            data=data,
            feature_names=feature_names,
            output_names=[label_key],
        )

    def display_label(self, raw_label: str) -> str:
        """Return a user-friendly label name for UI display."""
        return self.display_map.get(raw_label, raw_label)

    def explain_attention(self, text: str):
        """Extract attention weights from the model."""
        # Tokenize
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=500)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
        
        # Get attentions: tuple of (batch, num_heads, seq_len, seq_len)
        # We take the last layer: outputs.attentions[-1]
        # Squeeze batch: [num_heads, seq_len, seq_len]
        last_layer_attentions = outputs.attentions[-1][0]
        
        # Get attention from [CLS] token (index 0) to all other tokens for EACH head
        # Shape: [num_heads, seq_len]
        cls_attention_per_head = last_layer_attentions[:, 0, :].cpu().numpy()
        
        # Average across heads: [seq_len]
        avg_attention = cls_attention_per_head.mean(axis=0)
        
        # Get raw tokens
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return tokens, avg_attention, cls_attention_per_head

    def generate_attention_html(self, tokens: List[str], values: np.ndarray) -> str:
        """Generate HTML heatmap for attention values."""
        # Normalize values
        if len(values) == 0:
            return ""
        
        # Filter out special tokens for the visualization if needed, 
        # but keeping them ensures alignment. We'll just style them.
        
        max_val = values.max()
        if max_val == 0:
            max_val = 1.0
            
        html_parts = []
        for token, val in zip(tokens, values):
            # Handle RoBERTa spacing
            prefix = " " if "Ġ" in token else ""
            clean_token = token.replace("Ġ", "")
            
            # Skip rendering special tokens to make it look like natural text
            if clean_token in ["<s>", "</s>", "<pad>"]:
                continue
                
            # Calculate intensity
            # Using power < 1 to boost visibility of lower attention weights
            intensity = (val / max_val) ** 0.5 
            
            # Blue color with varying opacity
            style = f"background-color: rgba(0, 138, 255, {intensity}); color: black; border-radius: 2px; padding: 0 1px;"
            html_parts.append(f"<span style='{style}'>{prefix}{clean_token}</span>")
            
        return f"<div style='line-height: 1.6; font-family: sans-serif;'>{''.join(html_parts)}</div>"


@lru_cache(maxsize=1)
def get_detector(model_name: str = MODEL_NAME) -> BiasDetector:
    """Cached accessor to avoid reloading weights in Streamlit reruns."""
    return BiasDetector(model_name=model_name)
