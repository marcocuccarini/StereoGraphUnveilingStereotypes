import sys
import ollama
import pandas as pd
from pathlib import Path
from classes.llms_classes import *
from classes.prompt_classes import *
from classes.dataset_classes import TextDataset
from classes.matrix_creation_classes import *
from prompt import *
import re
import random
import nltk
import json

# Check NLTK punkt
try:
    nltk.data.find('tokenizers/punkt')
    print('punkt 🫡')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

from classes.class_evaluation import *

# ==========================
# CONFIGURABLE PARAMETERS
# ==========================
MODELS = ['llama3.1', 'mistral:7b', 'gemma2:9b']
EXPERIMENTS = ['baseline', 'kg']
sample_test = 2         # Number of examples to test (e.g., 513 for full)
sample_size = 20        # Number of triples to provide in KG prompt
LENGUAGE = "ITA"        # Language for KG context

# ==========================
# DATASET LOADING
# ==========================
text_dataset = pd.read_csv("../../data/test_graph_creation/stereo_test.csv", sep=",", on_bad_lines='skip')

# ==========================
# INIT OLLAMA SERVER
# ==========================
ollama_server = OllamaServer(ollama)
available_models = ollama_server.get_models_list()
print("Available Models:", available_models)

# Store results for all models
all_results = {}

# ==========================
# MAIN LOOP FOR EACH MODEL
# ==========================
for model_name in MODELS:
    print(f"\n=========== Running pipeline for MODEL: {model_name} ===========\n")
    all_results[model_name] = {}

    # Download model if needed
    ollama_server.download_model_if_not_exists(model_name)

    # Initialize chat
    chat = OllamaChat(server=ollama_server, model=model_name)

    # ========== BASELINE APPROACH ==========
    print(f"\n--- Testing BASELINE approach with model: {model_name} ---")
    response_baseline = []

    for i in range(len(text_dataset['tweet'][:sample_test])):
        post_text = text_dataset['tweet'][i]
        implied_statement = text_dataset['annotazione'][i]
        prompt2_baseline = costruisci_prompt3_mistrall(post_text)

        try:
            response = chat.send_prompt(prompt2_baseline, prompt_uuid="1", use_history=False)
            response_text_baseline = response.raw_text.strip()
        except Exception as e:
            response_text_baseline = f"[Error during prompt generation: {e}]"

        response_baseline.append(response_text_baseline)

        print("################## RESPONSE #####################\n")
        print("IMPLICIT STATEMENT PREDICTED:")
        print(response_text_baseline)
        print("\nIMPLICIT STATEMENT DATASET:")
        print(implied_statement)
        print("================================================\n")

    all_results[model_name]['baseline'] = response_baseline

    # ========== KG APPROACH ==========
    print(f"\n--- Testing KG approach with model: {model_name} ---")
    response_kg = []
    kg_samples = []

    for i in range(len(text_dataset['tweet'][:sample_test])):
        post_text = text_dataset['tweet'][i]
        implied_statement = text_dataset['annotazione'][i]

        context = retrieve_context_from_graph(text_dataset, post_text, i, sample_size, LENGUAGE=LENGUAGE)
        prompt2_kg = costruisci_prompt3_mistrall(post_text, context)

        try:
            response = chat.send_prompt(prompt2_kg, prompt_uuid="1", use_history=False)
            response_text_kg = response.raw_text.strip()
        except Exception as e:
            response_text_kg = f"[Errore durante la generazione della risposta: {e}]"

        response_kg.append(response_text_kg)
        kg_samples.append(context)

    all_results[model_name]['kg'] = response_kg
    all_results[model_name]['kg_samples'] = kg_samples

    # ========== EVALUATION ==========
    print(f"\n================ EVALUATION FOR MODEL: {model_name} ================\n")

    # Retrieve stored responses
    true_statements = text_dataset['annotazione'][:sample_test]
    sentence_base = list(zip(true_statements, response_baseline))
    sentence_kg = list(zip(true_statements, response_kg))

    print("----------- BASELINE RESULTS -----------")
    results_baseline = evaluate_all_similarities(sentence_base, chat)
    for method, result in results_baseline.items():
        print(f"{method.upper()} scores: {result['scores']}")
        print(f"{method.upper()} average: {result['average']:.4f}")
    print("\n")

    print("----------- KG RESULTS -----------")
    results_kg = evaluate_all_similarities(sentence_kg, chat)
    for method, result in results_kg.items():
        print(f"{method.upper()} scores: {result['scores']}")
        print(f"{method.upper()} average: {result['average']:.4f}")
    print("\n")

    # ========== JSON EXPORT ==========
    print(f"\n>>> Exporting results for MODEL: {model_name}")
    combined_data = []
    for idx, (implied, baseline, kg, kg_sample, post) in enumerate(zip(
            true_statements, response_baseline, response_kg, kg_samples, text_dataset['tweet'][:sample_test])):

        combined_data.append({
            "post": post,
            "implied_statement": implied,
            "baseline_response": baseline,
            "kg_response": kg,
            "bleu_similarity_baseline": str(results_baseline['bleu']['scores'][idx]),
            "bert_similarity_baseline": str(results_baseline['embedding']['scores'][idx]),
            "bleu_similarity_kg": str(results_kg['bleu']['scores'][idx]),
            "bert_similarity_kg": str(results_kg['embedding']['scores'][idx])
        })

    file_name = f"../results_evaluation/results/prediction_results_{model_name.replace(':', '_')}_prompt3_{LENGUAGE}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON file '{file_name}' created successfully.\n")
