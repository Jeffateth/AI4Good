# The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses

<!-- =======================================================
     LOGO BLOCK — ETH Zurich | Health Ethics & Policy Lab | NeurIPS
     ======================================================= -->
<p align="center">
  <img src="docs/assets/eth_logo_kurz_pos.svg" alt="ETH Zurich" width="200" style="display:inline-block; margin: 0 20px;"/>
  <img src="docs/assets/ETH_Health_Ethics_And_Policy_Lab_Logo.png" alt="Health Ethics & Policy Lab" width="200" style="display:inline-block; margin: 0 20px;"/>
  <img src="docs/assets/NeurIPS-logo.svg" alt="NeurIPS 2025" width="160" style="display:inline-block; margin: 0 20px;"/>
</p>

---

**Official repository** for our NeurIPS 2025 *GenAI4Health* workshop paper evaluating large language models on the understandability and empathy in medical diagnosis delivery.

**Authors:** Jianzhou Yao¹†, Shunchang Liu²†, Guillaume Drui², Rikard Pettersson¹, Alessandro Blasimme³, Sara Kijewski³*  
¹ Dept. of Chemistry and Applied Biosciences, ETH Zurich  
² Dept. of Computer Science, ETH Zurich  
³ Dept. of Health Sciences and Technology, ETH Zurich  
† Equal contribution | * Corresponding author: [sara.kijewski@hest.ethz.ch](mailto:sara.kijewski@hest.ethz.ch)

---

## Overview

We evaluate GPT-4o and Claude-3.7 on 156 synthetic medical diagnostic scenarios across diverse patient demographics, assessing:

- **Understandability:** using 5 standard readability metrics (Flesch-Kincaid, SMOG, Gunning Fog, Coleman-Liau, Dale-Chall)  
- **Empathy:** Affective and cognitive dimensions via LLM-as-a-judge and human ratings

**Scenario Design**
- 156 diagnostic scenarios combining 4 conditions × 3 education levels × 4 age groups × 2 genders × 3 regions

**Highlights**
- Texts exceed 6th–8th grade readability standards for health materials  
- Consistent empathy bias by condition, age, and education  
- Evaluator bias identified in LLM-as-a-judge settings  

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Jeffateth/Biased_Oracle.git
cd Biased_Oracle
conda env create -f environment.yml
conda activate AI4Good

# Generate figures
python src/readability_metrics_plot.py
python src/plot_empathy.py
Precomputed results are included; regenerating outputs requires OpenAI and Anthropic API keys.
```
## Repository Structure
```bash
Code kopieren
Biased_Oracle/
├── data/          # Prompts, model outputs, ratings
├── notebooks/     # Data collection & scoring
├── src/           # Analysis & plotting scripts
└── outputs/       # Figures and statistical results
```


## Citation
```bibtex
@inproceedings{
  yao2025the,
  title={The Biased Oracle: Assessing {LLM}s{\textquoteright} Understandability and Empathy in Medical Diagnoses},
  author={Jianzhou Yao and Shunchang Liu and Guillaume Drui and Rikard Pettersson and Alessandro Blasimme and Sara Kijewski},
  booktitle={The Second Workshop on GenAI for Health: Potential, Trust, and Policy Compliance},
  year={2025},
  url={https://openreview.net/forum?id=mhtDi2d4ZC}
}
```
## Notes
All data are synthetic; no real patient information is used.
Paper and workshop: OpenReview · GenAI4Health

## License
CC BY 4.0 – Free to use with attribution

For technical questions: Open a GitHub issue

<p align="center" style="font-size: 0.85em; color: gray;"> Logos used in accordance with institutional guidelines.<br/> © 2025 ETH Zürich · Health Ethics & Policy Lab · NeurIPS Foundation </p> ```
