# 🤖 AI Chatbot Evaluation Framework

## The Growing Need for Reliable AI Chatbot Information: Investigating Accuracy, Authenticity, and Up-to-dateness in Conversational AI Systems

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Course](https://img.shields.io/badge/Course-CSE%20533-red.svg)](#)

> **Course:** CSE 533 – Machine Learning  
> **Supervisor:** Dr. Mohammad Ashrafuzzaman Khan [AzK]  
> **Authors:** MD Raihan Khan (24 25 311 650) & Zannatul Islam Proma (24 25 308 650)  
> **Portfolio:** [raihankhan.info](https://raihankhan.info)

---

## 📌 Project Overview

Modern AI Chatbots (ChatGPT, Gemini, Grok, Claude, DeepSeek) are increasingly used as primary information sources. However, their reliability regarding **accuracy**, **authenticity**, and **up-to-dateness** remains largely unverified and potentially inconsistent.

This project develops a systematic evaluation framework to benchmark and compare the information quality of five leading AI chatbots across multiple domains and query types.

### Chatbots Evaluated

| Chatbot | Provider |
|---------|----------|
| ChatGPT | OpenAI |
| Gemini | Google |
| Grok | xAI |
| Claude | Anthropic |
| DeepSeek | DeepSeek AI |

---

## 🏗️ Project Structure

```
AI-ChatBOT-Evaluation/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── configs/
│   └── config.yaml              # Central configuration
├── data/
│   ├── queries/
│   │   └── evaluation_queries.json   # Curated query dataset
│   ├── responses/                    # Chatbot responses (generated)
│   ├── ground_truth/                 # Ground truth data (generated)
│   └── results/                      # Evaluation results (generated)
├── src/
│   ├── __init__.py
│   ├── query_manager.py         # Query dataset management
│   ├── chatbot_interface.py     # Chatbot API interaction layer
│   ├── ground_truth_collector.py # Ground truth data collection
│   ├── nlp_evaluator.py         # NLP-based evaluation engine
│   ├── scorer.py                # Scoring and metrics computation
│   ├── statistical_analysis.py  # Statistical tests and analysis
│   └── visualizer.py            # Charts, plots, and visual reports
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_evaluation_pipeline.ipynb
│   └── 03_analysis_and_visualization.ipynb
├── tests/
│   └── test_evaluator.py
├── main.py                      # Main execution entry point
└── run_evaluation.py            # Full pipeline runner
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Raihan-Khan-CS/AI-ChatBOT-Evaluation.git
cd AI-ChatBOT-Evaluation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure API Keys (Optional)

Copy the config template and add your API keys for automated chatbot querying:

```bash
cp configs/config.yaml configs/config_local.yaml
# Edit configs/config_local.yaml with your API keys
```

### 4. Run the Evaluation Pipeline

```bash
# Run the complete pipeline
python main.py --mode full

# Run individual stages
python main.py --mode collect    # Collect responses only
python main.py --mode evaluate   # Evaluate collected responses
python main.py --mode analyze    # Statistical analysis & visualization
```

---

## 📊 Methodology

### Evaluation Dimensions

| Dimension | Description | Scoring |
|-----------|-------------|---------|
| **Accuracy** | Factual alignment with ground truth | 0–5 Likert scale |
| **Authenticity** | Consistency, logical reasoning, absence of fabrication | 0–5 Likert scale |
| **Up-to-dateness** | Currency of information relative to ground truth timestamps | Binary + recency score |

### NLP Techniques Used

- **Keyword Matching** — TF-IDF based overlap scoring
- **Named Entity Recognition (NER)** — Entity extraction and comparison via SpaCy
- **Sentence Similarity** — Semantic similarity using Sentence-BERT embeddings
- **Fact Extraction** — Claim-level decomposition and verification

### Query Categories

| Domain | Query Types |
|--------|-------------|
| Science | Factual, Definitional |
| History | Factual, Comparative |
| Current Events | Recent events, Up-to-dateness |
| Technology | Factual, Comparative |
| General Knowledge | Factual, Definitional |

### Statistical Analysis

- **Descriptive Statistics** — Mean, median, std deviation per chatbot
- **ANOVA** — Multi-group comparison across chatbots
- **Pairwise t-tests** — With Bonferroni correction
- **Correlation Analysis** — Between evaluation dimensions
- **Visualization** — Bar charts, radar plots, heatmaps, box plots

---

## 📈 Expected Outputs

After running the pipeline, results are saved in `data/results/`:

- `evaluation_scores.csv` — Per-query scores for each chatbot
- `summary_statistics.csv` — Aggregate performance metrics
- `statistical_tests.json` — ANOVA and t-test results
- `figures/` — All generated charts and plots

---

## 📚 References

### Research Papers

1. Hettige & Karunananda (2019). *A survey on evaluation methods for chatbots.* ACM.
2. Wang et al. (2024). *Factuality of large language models: A survey.* arXiv:2402.02420.
3. Huang et al. (2023). *A survey on hallucination in large language models.* arXiv:2311.05232.
4. Augenstein et al. (2024). *Factuality challenges in the era of LLMs.* Nature Machine Intelligence.
5. Rajpurkar et al. (2016). *SQuAD: 100,000+ questions for machine comprehension.* arXiv:1606.05250.
6. Thorne et al. (2018). *FEVER: Fact extraction and VERification dataset.* arXiv:1803.05355.

### Tools & Libraries

- [SpaCy](https://spacy.io/) — NER and NLP pipeline
- [Sentence-Transformers](https://sbert.net/) — Semantic similarity
- [NLTK](https://www.nltk.org/) — Text processing
- [Scikit-learn](https://scikit-learn.org/) — TF-IDF and ML utilities
- [SciPy](https://scipy.org/) — Statistical testing
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — Visualization

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Authors

| | Name | ID | Email |
|---|------|-----|-------|
| 👨‍💻 | MD Raihan Khan | 24 25 311 650 | raihan.khan.242@northsouth.edu |
| 👩‍💻 | Zannatul Islam Proma | 24 25 308 650 | zannatul.proma.242@northsouth.edu |

---

<p align="center">
  <b>North South University — Department of ECE</b><br>
  CSE 533: Machine Learning — Project iDEA
</p>
