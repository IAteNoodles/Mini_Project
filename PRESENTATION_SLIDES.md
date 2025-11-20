# VISVESVARAYA TECHNOLOGICAL UNIVERSITY
**Belagavi - 590 018**

![VTU Logo Placeholder](vtu_logo.png)

**Subject Code: [CODE] – Mini Project**

# Presentation Slides: Media Bias Analysis System

## Slide 1: Title Slide

**MEDIA BIAS ANALYSIS SYSTEM**

*(Mini Project - 18CSMP68)*

**Presented By:**
*   [Student Name 1] ([USN])
*   [Student Name 2] ([USN])
*   [Student Name 3] ([USN])
*   [Student Name 4] ([USN])

**Guided By:**
*   **Prof. [Name]**
*   Assistant Professor
*   Dept. of CSE (Data Science)

**Acharya Institute of Technology**
*Visvesvaraya Technological University*

---

## Slide 2: Outline

*   Introduction
*   Problem Statement
*   Objectives
*   Literature Survey
*   System Requirements
*   System Architecture
*   Methodology
*   System Testing
*   Results & Discussion
*   Conclusion & Future Scope

---

## Slide 3: Introduction

*   **Overview**: Media bias is the polarized presentation of news, often subtle and manipulative.
*   **The Need**: In the era of "Fake News", detecting *bias* (slanted truth) is as important as detecting falsehoods.
*   **Our Solution**: An AI-powered web app that detects bias and explains *why* using Explainable AI (XAI).

---

## Slide 4: Problem Statement

*   **Subtlety**: Bias lies in word choice ("regime" vs "government"), not just facts.
*   **Volume**: Too much news for manual fact-checking.
*   **Trust**: Existing AI tools are "Black Boxes" – they give a verdict but no reason.
*   **Goal**: Build a **Transparent** and **Explainable** detection system.

---

## Slide 5: Objectives

1.  **Detect**: Classify news as "Biased" or "Non-Biased" using **RoBERTa**.
2.  **Explain**: Highlight biased words using **SHAP** and **LIME**.
3.  **Narrate**: Generate natural language summaries using **Generative AI (Granite)**.
4.  **Deploy**: Provide a user-friendly **React** interface.

---

## Slide 6: Literature Survey

| Author (Year) | Method | Limitation |
| :--- | :--- | :--- |
| **Hamborg et al. (2019)** | Matrix Factorization | Limited to political bias; no explanations. |
| **Spinde et al. (2021)** | LSTM / BERT | High accuracy but "Black Box" nature. |
| **Ribeiro et al. (2016)** | LIME | Good local explanation, but computationally expensive. |

*   **Our Approach**: Combines **RoBERTa** (Accuracy) + **SHAP** (Global Explanation) + **LLM** (Readability).

---

## Slide 7: System Requirements

**Hardware:**
*   Processor: Intel i5 / Ryzen 5
*   RAM: 8GB+
*   GPU: NVIDIA GTX 1650 (Recommended for training)

**Software:**
*   **Frontend**: React.js
*   **Backend**: FastAPI (Python)
*   **ML Models**: PyTorch, Transformers, SHAP
*   **LLM**: Ollama (Granite 4:3b)

---

## Slide 8: System Architecture

*(Placeholder for Architecture Diagram)*

1.  **User Input**: Text/URL via React UI.
2.  **API Gateway**: FastAPI routes request.
3.  **Model Inference**: RoBERTa predicts probability.
4.  **XAI Engine**: SHAP calculates feature importance.
5.  **LLM Service**: Generates text explanation.
6.  **Output**: Verdict + Heatmap + Summary.

---

## Slide 9: Methodology

1.  **Preprocessing**:
    *   Cleaning (HTML/Special chars).
    *   Tokenization (RoBERTa Tokenizer).
    *   Padding/Truncation (512 tokens).
2.  **Training**:
    *   Fine-tuned `roberta-base` on MBIC dataset.
    *   Optimizer: AdamW.
3.  **Explainability**:
    *   **SHAP**: Assigns "contribution scores" to words.
    *   **LIME**: Perturbs text to find local decision boundaries.

---

## Slide 10: System Testing

| Test Case | Input Snippet | Expected | Actual | Result |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | "The city council met at 10 AM." | Non-Biased | Non-Biased | **PASS** |
| **TC-02** | "The radical left is destroying us." | Biased | Biased | **PASS** |
| **TC-03** | "Corrupt regime oppresses people." | Biased | Biased | **PASS** |

---

## Slide 11: Results & Discussion

*   **Performance**:
    *   **Accuracy**: ~85%
    *   **F1-Score**: 0.85
*   **Inference Speed**: < 1 second per article.
*   **Visualization**: Heatmaps successfully identify subjective adjectives (e.g., "radical", "disaster").

---

## Slide 12: Snapshots

*(Placeholder for UI Screenshots)*

*   **Home Page**: Simple text area for input.
*   **Results Page**:
    *   **Gauge Chart**: Shows Bias Probability.
    *   **Text Highlight**: Red color for biased words.
    *   **AI Summary**: "This article uses emotional language..."

---

## Slide 13: Conclusion

*   The system effectively bridges the gap between **High-Performance AI** and **Human Understandability**.
*   It empowers users to consume news critically by showing *where* and *why* bias exists.

---

## Slide 14: Future Scope

*   **Multilingual Support**: Hindi, Kannada, Spanish.
*   **Browser Extension**: Real-time analysis while browsing.
*   **Video/Audio Analysis**: For TV news debates.

---

## Slide 15: References

1.  Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach".
2.  Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP).
3.  Hamborg et al., "Automated identification of media bias".

---

# THANK YOU


---

## Content

1.  Abstract
2.  Problem Statement
3.  Introduction
4.  Objective
5.  Literature Survey
6.  System Architecture / Flow Diagram
7.  Methodology
8.  System Testing
9.  Results Discussion
10. Demonstration of Prototype
11. Future Works
12. Conclusion
13. References

---

## Abstract

*   Traditional media literacy tools fail against the speed and subtlety of modern digital bias.
*   AI-powered detection using **RoBERTa** (Transformer) with **Attention Mechanism** and **Explainable AI (XAI)** via a React/FastAPI web interface.
*   Provides protection with high accuracy, safeguarding users from misinformation and manipulation.

---

## Problem Statement

*   Media bias is a widespread issue impacting public opinion globally.
*   Traditional manual fact-checking often fails to detect subtle linguistic framing and tone.
*   Need for intelligent, adaptable, and transparent detection solutions.

---

## Introduction

*   Biased news exploits reader trust via deceptive framing and word choice.
*   Increasing sophistication of polarization requires advanced defense.
*   **Motivation**: Protect users from manipulation and echo chambers.
*   Valid and urgent in the current digital era. Scope includes individuals, researchers, and media organizations.

---

## Objective

*   Build an effective, AI-powered media bias detection system.
*   Ensure high accuracy and low false positives in bias classification.
*   Provide **explainable insights** (Why is it biased?) using SHAP and LIME.

---

## Literature Survey

| Sl No. | Citations | Methodologies | Research Gaps |
| :--- | :--- | :--- | :--- |
| 01 | *Hamborg et al. (2019)* - "Automated identification of media bias in news articles" | Matrix factorization and word embeddings to detect slanted framing. | Focuses on political bias only; lacks user-friendly explanations for *why* a text is biased. |
| 02 | *Spinde et al. (2021)* - "Media Bias Detection using Deep Learning" | Compares LSTM vs. BERT for bias classification. | High accuracy but operates as a "black box" with no transparency for the end-user. |
| 03 | *Ribeiro et al. (2016)* - "Why Should I Trust You?" | LIME for local interpretability of black-box models. | General-purpose XAI; not specifically optimized for the nuances of media bias detection. |

---

## System Architecture / Flow Diagram

![System Architecture Diagram](architecture_diagram.png)

*   **Input**: News Article Text/URL.
*   **Preprocessing**: Tokenization, Cleaning.
*   **Model**: RoBERTa Transformer.
*   **Explanation**: SHAP/LIME/Attention.
*   **Output**: Bias Verdict + Narrative Report.

---

## Methodology

*   Collect **MBIC (Media Bias Identification Benchmark)** dataset for training.
*   Preprocess article text, build vocabulary (RoBERTa Tokenizer).
*   Train deep learning model with class balancing and regularization.
*   Deploy **FastAPI**-based backend and **React** frontend for real-time detection.
*   Use **Ollama (Granite 4:3b)** for generating natural language explanations.

---

## System Testing

| Sample Article Input | Classification | Confidence Score |
| :--- | :--- | :--- |
| "The radical left's agenda is destroying our traditional values and ruining the economy." | **Biased** | **98.2%** |
| "The city council met yesterday to discuss the new budget proposal for the upcoming fiscal year." | **Non-Biased** | **12.4%** |
| "The corrupt regime continues to oppress its citizens without mercy, ignoring international calls for peace." | **Biased** | **96.5%** |
| "Scientists published a study linking high sugar intake to increased health risks in adults." | **Non-Biased** | **8.1%** |
| "An absolute disaster of a policy that will surely lead to the downfall of our great nation." | **Biased** | **94.8%** |

---

## Results Discussion

![Confusion Matrix / Accuracy Graph](results_graph.png)

*   The model achieves **~85% Accuracy** on the test set.
*   **SHAP Heatmaps** successfully highlight subjective adjectives (e.g., "radical", "corrupt").
*   Inference time optimized to **<1 second** using CUDA acceleration.

---

## Demonstration of Prototype

![Screenshot of React Frontend](demo_screenshot.png)

*   **User Interface**: Clean input box for text.
*   **Analysis View**: Real-time "Biased" vs "Non-Biased" gauge.
*   **Explainability**: "Explain with AI" button generates a readable report.

---

## Future Works

*   Expand dataset to include **multi-language support** (Hindi, Spanish).
*   Integrate with **browser extensions** for real-time analysis on news sites.
*   Improve model with larger **LLM-based** architectures for nuance detection.

---

## Conclusion

*   **Media Bias Analysis System** provides robust, real-time bias detection.
*   Harnesses advanced **AI and XAI** to safeguard users effectively.
*   Promotes critical thinking and media literacy in the digital age.

---

## References

1.  Liu, Y., et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv preprint arXiv:1907.11692 (2019).
2.  Lundberg, S. M., & Lee, S. I. "A Unified Approach to Interpreting Model Predictions." NeurIPS (2017).
3.  Hamborg, F., Donnay, K., & Gipp, B. "Automated identification of media bias in news articles." International Journal on Digital Libraries (2019).

---

**THANK YOU**
