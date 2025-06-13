import sys
print(sys.version)
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

try:
    nltk.data.find('tokenizers/punkt')
    print('punkt 🫡')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

from class_evaluation import *


LENGUAGE="ITA"   # define the langauge of the test, "ITA" for italian, "ENG" for english 

text_dataset=pd.read_csv("../data/"+str(LENGUAGE)+"/test_set.csv", sep=";", on_bad_lines='skip')

#few_shot, dataframe_prompt, samples few_shot

prompt=PromptCreation(True, "data_prompt_contruction/test_prompts.csv","data_prompt_contruction/samples_fewshot.csv")

resulting_prompt=prompt.prompt_creation()


# Init the ollama server
ollama_server = OllamaServer(ollama)

# Check models that have already been downloaded
models = ollama_server.get_models_list()
print("Available Models:", models)

#MODELS = ['llama3.1']
#MODELS = ['mistral:7b']
#MODELS = ['gemma2:9b']
MODELS = ['llama3.1', 'mistral:7b', 'gemma2:9b']

EXPERIMENTS = ['baseline', 'kg']          

sample_test = 513 #quanti esempi testare 
sample_size= 20 # Numero massimo di esempi froniti dal kg (funzione di cui ne parlava Lia)
all_results = {}

#---------  SET UP OLLAMA + SELEZIONE MODELLO ---------------------

for MODEL in MODELS:
    all_results[MODEL] = {}

    # Download the model to use for experiences
    models = ollama_server.download_model_if_not_exists(MODEL)

    # Initialize chat with a specific model
    chat = OllamaChat(server=ollama_server, model=MODEL)

# --------------- APPROCCIO BASELINE -----------------------------
print(f"\n--- Test dell'approccio BASELINE con modello: {MODEL} ---")

response_baseline = []

# Create the object for the creation of the support matrix
for i in range(len(text_dataset['post'][:sample_test])):
    post_text = text_dataset['post'][i]
    implied_statement = text_dataset['implied_statement'][i]

    # Format the prompt
    #prompt = resulting_prompt.to_list()[0].format(post_text)

    prompt2_baseline = costruisci_prompt3_mistrall(post_text)

    
    try:
        response = chat.send_prompt(prompt2_baseline, prompt_uuid="1", use_history=False) #, stream=True)
        response_text_baseline = response.raw_text.strip()
    except Exception as e:
        response_text_baseline = f"[Error during prompt generation: {e}]"

    # Store response
    response_baseline.append(response_text_baseline)

    #Print response and reference
    print("################## RESPONSE #####################\n")
    print("IMPLICIT STATEMENT PREDICTED:")
    print(response_text_baseline)
    print("\nIMPLICIT STATEMENT DATASET:")
    print(implied_statement)
    print("================================================\n")

#store model-specific baseline responses
all_results[MODEL]['baseline'] = response_baseline

# --------------- APPROCCIO CON KG -------------------------------------------
# Ora il ciclo con integrazione del contesto RDF
print(f"\n--- Test dell'approccio KG con modello: {MODEL} ---")
response_kg = []

for i in range(len(text_dataset['post'][:sample_test])):
    post_text = text_dataset['post'][i]
    implied_statement = text_dataset['implied_statement'][i]

    # Recupera contesto dal grafo RDF
    context = retrieve_context_from_graph(text_dataset, post_text, i, sample_size, LENGUAGE)

    # Costruisci il prompt con contesto
    prompt2 = costruisci_prompt3_mistrall(post_text, context)


    try:
        response = chat.send_prompt(prompt2, prompt_uuid="1", use_history=False) #, stream=True)
        response_text_kg = response.raw_text.strip()
    except Exception as e:
        response_text_kg = f"[Errore durante la generazione della risposta: {e}]"

    #store resp
    response_kg.append(response_text_kg)

    #print("################## RESPONSE #####################")
    #print("IMPLICIT STATEMENT PREDICTED:")
    #print(response_text_kg)
    #print("\nIMPLICIT STATEMENT DATASET:")
    #print(implied_statement)
    #print("================================================\n")

#store response from kg approach    
all_results[MODEL]['kg'] = response_kg
for MODEL in MODELS:
    print(f"\n================ EVALUATION FOR MODEL: {MODEL} ================\n")

    # Retrieve stored responses
    response_baseline = all_results[MODEL]['baseline']
    response_kg = all_results[MODEL]['kg']

    # Ensure alignment with sample_test
    true_statements = text_dataset['implied_statement'][:sample_test]

    # Pair gold labels with predictions
    sentence_base = list(zip(true_statements, response_baseline))
    sentence_kg = list(zip(true_statements, response_kg))

    # --- Baseline Evaluation ---
    print("----------- BASELINE RESULTS -----------")
    results_baseline = evaluate_all_similarities(sentence_base, chat)
    for method, result in results_baseline.items():
        print(f"{method.upper()} scores: {result['scores']}")
        print(f"{method.upper()} average: {result['average']:.4f}")
    print("\n")

    # --- KG Evaluation ---
    print("----------- KG RESULTS -----------")
    results_kg = evaluate_all_similarities(sentence_kg, chat)
    for method, result in results_kg.items():
        print(f"{method.upper()} scores: {result['scores']}")
        print(f"{method.upper()} average: {result['average']:.4f}")
    print("\n")


    import json

for MODEL in MODELS:
    print(f"\n>>> Exporting results for MODEL: {MODEL}")

    # Retrieve predictions
    baseline_responses = all_results[MODEL]['baseline']
    kg_responses = all_results[MODEL]['kg']

    # Ensure ground-truth and KG context samples are aligned
    implied_statements = text_dataset['implied_statement'][:sample_test]

    # Pair gold labels with predictions for evaluation
    sentence_kg = list(zip(implied_statements, kg_responses))
    results_kg_eval = evaluate_all_similarities(sentence_kg, chat)

    # NOTE: you must prepare a list of KG context samples used for each item
    # For example, during the KG run:
    # all_results[MODEL]['kg_samples'] = list_of_kg_contexts
    kg_samples = all_results[MODEL].get('kg_samples', [""] * sample_test)

    # Construct data to export
    combined_data = []
    cont=0
    for implied, baseline, kg, kg_sample, post in zip(implied_statements, baseline_responses, kg_responses, kg_samples, text_dataset['post'][:sample_test]):
        combined_data.append({
            "post": post,   # <-- added this line to include original post text
            "implied_statement": implied,
            "baseline_response": baseline,
            "kg_response": kg,
            #"kg_sample": kg_sample,
            #"kg_sample_size": len(kg_sample.split("\n")) if isinstance(kg_sample, str) else 0,
            "bleu_similarity_baseline": str(results_baseline['bleu']['scores'][cont]),
            "bert_similarity_baseline": str(results_baseline['embedding']['scores'][cont]),
            "bleu_similarity_kg": str(results_kg_eval['bleu']['scores'][cont]),
            "bert_similarity_kg": str(results_kg_eval['embedding']['scores'][cont])
        })
        cont+=1


    # Save to JSON
    file_name = f"result/prediction_results_{MODEL.replace(':', '_')}_prompt3_"+str(LENGUAGE)+".json"
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON file '{file_name}' created successfully.")



