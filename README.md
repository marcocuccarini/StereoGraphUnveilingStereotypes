# StereoGraph

**"Unveiling Stereotypes: StereoGraph-Enhanced LLMs for Knowledge-Driven Stereotype Explanation"**  
Presented at **CLIC-it 2025**

StereoGraph is a framework designed to enhance Large Language Models (LLMs) with structured knowledge for the detection and explanation of stereotypes. This repository contains all the code, data, and documentation needed to reproduce the experiments from the paper.

---

## 🚀 Features

- Combines LLMs with knowledge graphs for stereotype explanation
- Supports models like **Gemma**, **LLaMA**, and **Mistral** (via [Ollama](https://ollama.ai))
- Human and automated evaluation modules
- Jupyter-compatible pipeline for analysis and experimentation

---

## 💠 Installation

### 1. Clone the repository
```bash
git clone https://github.com/marcocuccarini/StereoGraphUnveilingStereotypes.git
cd StereoGraphUnveilingStereotypes
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama
Follow the official installation instructions from [ollama.ai](https://ollama.ai).


### 4. Import the dataset 
From the following GitHub page: [Open-Stereotype-corpus](https://github.com/SodaMaremLo/Open-Stereotype-corpus) and import into the test_graph_creation, with the name "stereo_test.csv".
Rember to split the dataset using the samples (ids) present in the dataset test_set.csv". 

### 5. Run the code main.py
It save the results om script, result_evalaution results

### 6. Run the evalaute_perfroamnce.py update the json fil, adding the avarege results in the fon of thejson file



---

## 📂 Repository Structure

```
StereoGraph/
│
├── data/                              # Dataset folders
│   ├── error analysis/                # IDs and human annotations for error analysis
│   └── test e graph creation/         # Dataset and IDs used for graph creation and testing
│
├── human_evaluation/                  # Human evaluation results (error and similarity analyses)
│
├── kg/                                # Populated knowledge graph (ontology instances)
│
├── ontology/                          # Ontology schema used in experiments
│
├── scripts/                           # Scripts for pipeline execution and evaluation
│   ├── pipeline_implementation/       # Core pipeline components
│   │   ├── main.py                    # Script to run and test the model
│   │   └── prompt.py                  # List of possible prompts for evaluation
│   │
│   ├── results_evaluation/           # Evaluation outputs and utilities
│   │   └── evaluate_performance.py    # Extracts average similarity from result files
│
├── src/                               # SPARQL queries and related utilities
│
├── main.py                            # CLI entry point for the full pipeline
└── prompts.py                         # Prompt templates used in the evaluation

```

---

## 🚀 Running the Pipeline

Run the experiment using either of the following:

### Option 1: Python script
```bash
python main.py
```

### Option 2: Jupyter Notebook
Open `main.ipynb` and run all cells.

The system runs experiments using three LLMs: **Gemma**, **LLaMA**, and **Mistral**. It processes a triple-stereotype dataset and can be extended to any model available via **Ollama**.

---

## 📚 Data

- `data/error analysis/`: IDs and human annotations used in error analysis
- `data/test e graph creation/`: Dataset used to create the knowledge graph and for testing
- `human_evaluation/`: Contains the results of human evaluations

---

## 🔗 Contribution

We welcome contributions! Please open issues or submit pull requests if you find bugs or want to improve the project.

---

## 📖 Citation

If you use this work, please cite the following paper:

> **Unveiling Stereotypes: StereoGraph-Enhanced LLMs for Knowledge-Driven Stereotype Explanation**  
> CLIC-it 2025

(Citation BibTeX coming soon)

---

## 🚫 License

This project is licensed under the MIT License. See `LICENSE` for details.

