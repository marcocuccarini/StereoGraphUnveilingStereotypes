# StereoGraph

This GitHub repository contains the experiments and resources used to reproduce the work titled:  
**"Unveiling Stereotypes: StereoGraph-Enhanced LLMs for Knowledge-Driven Stereotype Explanation"**  
Presented at CLIC-it 2025.

## Repository Structure

- **`data/`**  
  Contains all datasets. The full dataset is available via the provided link.  
  We include unique IDs that define the split between the test set and the graph construction set.

- **`human_evaluation/`**  
  Includes the results of the error analysis and similarity evaluations performed by human annotators.

- **`kg/`**  
  Contains the populated ontology built from posts included in the graph dataset.

- **`ontology/`**  
  Defines the structure of the ontology used in the experiments.

- **`src/`**  
  Includes the SPARQL queries used throughout the project.

- **`scripts/`**  
  Contains the code to run the full pipeline and perform both automated and human-based evaluations.

## Running the Pipeline

To run the full pipeline, execute either `main.py` or `main.ipynb`.  
The `prompts/` file (or directory) includes various prompts used for evaluation and testing.
