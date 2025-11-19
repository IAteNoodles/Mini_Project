# Presentation Slides Script
**Project:** Media Bias Analysis System
**Based on Template:** PhishGuard PPT

---

## Slide 1: Title Slide

**[Center Alignment]**

**VISVESVARAYA TECHNOLOGICAL UNIVERSITY**
Belagavi - 590 018

**(Logo Placeholder)**

**Subject Code: [CODE] – Mini Project**

# MEDIA BIAS ANALYSIS SYSTEM
**AI-Powered Bias Detection & Explanation**

**Presented By:**
*   [Student Name 1] ([USN])
*   [Student Name 2] ([USN])
*   [Student Name 3] ([USN])
*   [Student Name 4] ([USN])

**Under the Guidance of:**
*   **Prof. [Name]**
*   Assistant Professor

**DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING (DATA SCIENCE)**
**ACHARYA INSTITUTE OF TECHNOLOGY**
**November 19, 2025**

---

## Slide 2: Content

*   Abstract
*   Problem Statement
*   Introduction
*   Objective
*   Literature Survey
*   System Architecture / Flow Diagram
*   Methodology
*   System Testing
*   Results Discussion
*   Demonstration of Prototype
*   Future Works
*   Conclusion
*   References

---

## Slide 3: Abstract

*   **The Challenge**: Traditional fact-checking fails against the speed and subtlety of modern digital media bias.
*   **Our Solution**: An AI-powered detection system using **RoBERTa** (Deep Learning) with **Attention Mechanisms** and **Explainable AI (XAI)**.
*   **Key Features**:
    *   Real-time classification (Biased vs. Non-Biased).
    *   Visual heatmaps (SHAP/LIME) showing *why* text is biased.
    *   AI-generated narrative explanations via LLMs.
*   **Impact**: Safeguards users from manipulation and promotes media literacy.

---

## Slide 4: Problem Statement

*   **Information Overload**: Millions of articles are published daily; manual verification is impossible.
*   **Subtle Manipulation**: Bias isn't just "fake news"—it's framing, tone, and omission.
*   **Black Box AI**: Existing tools give a score but don't explain the "why," leading to user distrust.
*   **Need**: A system that is **Intelligent**, **Transparent**, and **User-Friendly**.

---

## Slide 5: Introduction

*   **Context**: Media bias polarizes society and distorts public perception of reality.
*   **Sophistication**: Modern bias is often linguistic (e.g., "regime" vs. "government"), requiring advanced NLP to detect.
*   **Motivation**: To empower readers with a "digital truth lens."
*   **Scope**: Focuses on English-language political and social news articles.

---

## Slide 6: Objective

*   **Build**: A robust, full-stack web application for bias analysis.
*   **Detect**: Achieve high accuracy using fine-tuned Transformer models (RoBERTa).
*   **Explain**: Integrate XAI (SHAP, LIME) to visualize feature importance.
*   **Synthesize**: Use Generative AI to write human-readable bias reports.

---

## Slide 7: Literature Survey

| Citation | Methodology | Research Gap |
| :--- | :--- | :--- |
| **Hamborg et al. (2019)** | Matrix factorization for framing detection. | Limited to political topics; no user-facing explanations. |
| **Ribeiro et al. (2016)** | LIME for local model interpretability. | General purpose; not tuned for linguistic nuance in news. |
| **Spinde et al. (2021)** | Deep Learning (LSTM/BERT) for classification. | High accuracy but lacks transparency ("Black Box"). |

---

## Slide 8: System Architecture / Flow Diagram

**(Insert Architecture Diagram Here)**

*   **Frontend**: React.js (User Interface).
*   **Backend**: FastAPI (Model Serving).
*   **Core Model**: RoBERTa (Classification).
*   **XAI Engine**: SHAP & LIME (Feature Attribution).
*   **Narrative Engine**: Ollama/Granite (Text Generation).

---

## Slide 9: Methodology

1.  **Data Collection**: Aggregated datasets (MBIC, BASIL) containing labeled biased/neutral articles.
2.  **Preprocessing**: Tokenization, cleaning, and sequence truncation (512 tokens).
3.  **Training**: Fine-tuned `roberta-base` with cross-entropy loss and AdamW optimizer.
4.  **XAI Integration**: Implemented KernelSHAP and LimeTextExplainer for post-hoc analysis.
5.  **Deployment**: Containerized application with a React frontend for real-time interaction.

---

## Slide 10: System Testing

| Input Snippet | Classification | Confidence |
| :--- | :--- | :--- |
| "The radical agenda is destroying our values." | **Biased** | **98.2%** |
| "The committee met to discuss the budget." | **Legitimate** | **12.4%** |
| "An absolute disaster of a policy." | **Biased** | **94.8%** |
| "Scientists published the study results." | **Legitimate** | **8.1%** |
| "The corrupt regime oppresses its people." | **Biased** | **96.5%** |

---

## Slide 11: Results & Discussion

*   **Performance**: Model achieves ~85% F1-Score, significantly outperforming keyword-based baselines.
*   **Interpretability**: Heatmaps successfully identify subjective adjectives (e.g., "radical", "corrupt") as bias triggers.
*   **User Feedback**: The "Explain with AI" feature (LLM narrative) was rated highly for making technical data understandable.

---

## Slide 12: Demonstration of Prototype

**(Insert Screenshots of the Tool)**

*   **Input**: User pastes text.
*   **Analysis**: System processes in <1 second.
*   **Output**:
    *   **Verdict**: "Biased" (Red) / "Non-Biased" (Green).
    *   **Heatmap**: Visual highlights of biased words.
    *   **Report**: AI-written summary of the findings.

---

## Slide 13: Future Works

*   **Multi-Language**: Support for Hindi, Kannada, and other regional languages.
*   **Browser Extension**: Analyze news directly on sites like Times of India or CNN.
*   **Video/Audio**: Analyze bias in TV news debates using Speech-to-Text.
*   **Crowdsourcing**: Allow users to flag incorrect predictions to retrain the model.

---

## Slide 14: Conclusion

*   **Summary**: We successfully built an end-to-end system for detecting and explaining media bias.
*   **Innovation**: The combination of **Deep Learning** (RoBERTa) and **Generative AI** (LLM Explanations) sets this project apart.
*   **Social Good**: Provides a crucial tool for fighting misinformation and promoting critical thinking.

---

## Slide 15: References

1.  Liu, Y., et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." (2019).
2.  Lundberg, S. M., & Lee, S. I. "A Unified Approach to Interpreting Model Predictions." (2017).
3.  Hamborg, F., et al. "Automated identification of media bias in news articles." (2019).
4.  Vaswani, A., et al. "Attention Is All You Need." (2017).

---

**THANK YOU**
