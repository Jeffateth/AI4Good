# Empathy and Understandability: Assessing LLMs in Delivering Compassionate Medical Diagnoses

This repository contains the code and data for evaluating the performance of Large Language Models (LLMs) in delivering empathetic and understandable medical diagnostic explanations across diverse patient demographics.

## 📋 Overview

We developed a comprehensive evaluation framework to assess how well commercial LLMs (GPT-4o and Claude-3.7) can deliver medical diagnoses with appropriate empathy and understandability across different patient populations. Our study reveals systematic demographic biases in AI-generated medical communications.

## 🎯 Key Findings

- **Medical diagnosis** is the strongest bias factor (p<0.0001) – Alzheimer’s receives highest empathy, heart disease lowest  
- **Education level** shows inverse relationship – medical degree holders receive 0.30–0.50 points lower empathy  
- **Age patterns** are rater-dependent – U-shaped empathy distribution emerges only with specific evaluators  
- **Cognitive empathy** remains stable across demographics while **affective empathy** varies substantially  
- **Critical methodological issue**: Poor inter-rater reliability between Claude and GPT evaluators

## 🔧 Setup

### Prerequisites

```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas
- numpy  
- matplotlib, seaborn
- textstat
- litellm
- scipy

### API Keys
Set up your API keys for:
- OpenAI (GPT-4o)
- Anthropic (Claude-3.7)

## 📁 Repository Structure

```
├── Code/
│   ├── Empathy_Score/          # Empathy evaluation pipeline
│   ├── Human/                  # Human annotation tools
│   ├── Judges_Score/           # LLM-as-judge evaluation
│   └── Plotting/               # Visualization scripts
├── Prompts_And_Response/       # Generated prompts and model responses
├── Ratings/                    # Empathy and understandability ratings
│   ├── Empathy/               # Empathy scores from different raters
│   └── Judges/                # EmotionQueen benchmark scores
├── Understandability/          # Readability analysis results
└── Scoring_Charts/            # Final visualizations and charts
```

## 🚀 Usage

### 1. Generate Diagnostic Scenarios
Open and run the Jupyter notebook that creates prompts:
```
Code/Empathy_Score/Prompt_and_Response_claude.ipynb
```

### 2. Collect Model Responses
Within the same or a similar notebook (e.g., `Prompt_and_Response_gpt.ipynb`), gather responses from GPT-4o and Claude-3.7.

### 3. Evaluate Understandability
Run the notebook:
```
Code/Understandability/empathy_ethics_understandability_analysis_rikard.ipynb
```
This calculates readability metrics and summarizes the output.

### 4. Assess Empathy
LLM-based empathy evaluation is demonstrated in:
```
Code/Empathy_Score/Prompt_and_Response_claude.ipynb
```
Human annotation can be found or adapted in:
```
Code/Human/human_annotation.ipynb
```

### 5. Generate Analysis
For statistical analysis and visualizations, use:
```
Code/Plotting/plot.ipynb
```
and other notebooks (e.g., `plot_EmotionQueen_diff.ipynb`) in the same directory.

## 📊 Evaluation Framework

### Two-Stage Assessment:

**Stage 1: Generation**
- 156 diagnostic scenarios combining:
  - Demographics: 3 ethnicities × 2 genders × 3 education levels
  - Medical conditions: Obesity, pancreatic cancer, Alzheimer's, heart disease
  - Age ranges: 8-85 years

**Stage 2: Evaluation**
- **Understandability**: 6 readability metrics (Flesch-Kincaid, SMOG, etc.)
- **Empathy**: Affective (emotional resonance) + Cognitive (perspective-taking)
- **Validation**: EmotionQueen benchmark + human annotation

## 🎨 Key Visualizations

- `scores_by_diagnosis.png` - Empathy bias across medical conditions
- `scores_by_education.png` - Education-level empathy patterns  
- `scores_by_age.png` - Age-related empathy variations
- `human_vs_LLM_1.png` - Human vs. AI evaluation comparison
- `framework.png` - Overall methodology diagram

## 📈 Results Summary

### Systematic Biases Identified:
1. **Medical Diagnosis** (strongest): Alzheimer's > Cancer > Obesity > Heart Disease
2. **Education Level**: High School > University > Medical Degree  
3. **Age** (rater-dependent): U-shaped pattern with higher empathy for children and elderly

### Methodological Insights:
- GPT consistently inflates own empathy ratings (+0.333 points)
- Claude deflates own empathy ratings (-0.256 points)
- Poor inter-rater reliability (r=-0.005 to 0.459)
- Human evaluators detect biases that LLMs miss

## 🔬 Reproducing Results

1. Run the complete pipeline:
```bash
python run_full_evaluation.py
```

2. Generate specific plots:
```bash
python Code/Plotting/plot_diagnosis_bias.py
python Code/Plotting/plot_education_bias.py
```

3. Statistical analysis:
```bash
python Code/statistical_analysis.py
```

## 👥 Authors

- **Shunchang Liu** - liushu@ethz.ch
- **Guillaume Drui** - gdrui01@ethz.ch  
- **Jianzhou Yao** - yaojia@ethz.ch
- **Rikard Pettersson** - rpettersson@student.ethz.ch

*Human-Centered AI for Social Good: Peace, Health, Climate, ETH Zurich*

## ⚠️ Ethical Considerations

This research reveals systematic biases in AI medical communication that could perpetuate healthcare disparities. Key concerns:

- Reduced empathy toward highly educated patients
- Differential treatment across medical conditions  
- Potential amplification of existing healthcare biases
- Need for human oversight in clinical deployment

## 🔄 Future Work

- Develop debiasing techniques for medical AI
- Improve evaluation methodology with better inter-rater reliability
- Expand demographic representation in training data
- Large-scale human validation studies
- Real-world clinical deployment guidelines

## 📞 Contact

For questions about the research or code, please contact the authors or open an issue in this repository.

---

**⚠️ Disclaimer**: This research is for academic purposes. Any deployment in clinical settings requires careful validation, bias mitigation, and human oversight to ensure patient safety and equitable care.
