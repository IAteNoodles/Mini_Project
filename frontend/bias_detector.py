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

if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


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

    def explain_lime(self, text: str, num_features: int = 10):
        """Generate a LIME explanation for the provided text."""
        return self._explainer.explain_instance(
            text_instance=text,
            classifier_fn=self.predict_proba,
            num_features=num_features,
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


@lru_cache(maxsize=1)
def get_detector(model_name: str = MODEL_NAME) -> BiasDetector:
    """Cached accessor to avoid reloading weights in Streamlit reruns."""
    return BiasDetector(model_name=model_name)
