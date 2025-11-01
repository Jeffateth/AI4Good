# Biased Oracle: Empathy & Understandability in LLM-Generated Medical Explanations

**Shunchang Liu**, **Guillaume Drui**, **Jianzhou Yao**, **Rikard Pettersson**  
Human-Centered AI for Social Good: Peace, Health, Climate, ETH Zürich

[Code](https://github.com/Jeffateth/Biased_Oracle) ·
<!-- Add when ready: [Paper](#) · [Dataset](#) -->
[License](../LICENSE) · [Cite](#bibtex)

![teaser](assets/teaser.png)

## Abstract
Large language models can tailor diagnostic explanations and tone to patient context, but do they remain **understandable** and **empathetic** across demographics? Biased Oracle evaluates GPT-4o and Claude-3.7 on 156 scenarios spanning demographics, education, conditions, and age. We combine readability metrics and LLM-as-a-Judge with human validation to expose systematic bias patterns.

## Key Findings
- **Diagnosis dominates empathy** (Alzheimer’s highest; heart disease lowest).
- **Education bias** (medical degree → lower affective empathy).
- **Age effects are rater-dependent** (U-shape visible with some evaluators).
- **LLM-as-Judge disagreement** (poor inter-rater reliability between models).

## Quick Start
```bash
pip install -r requirements.txt
# then follow Code/Empathy_Score/Prompt_and_Response_*.ipynb
