import json
import os
from class_evaluation import (
    sentence_bleu_score,
    sentence_similarity,
    sentence_rouge_l
)

def evaluate_and_save(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    evaluated_data = []

    # Initialize accumulators for averages
    total_bleu_baseline = 0.0
    total_bert_baseline = 0.0
    total_rouge_baseline = 0.0

    total_bleu_kg = 0.0
    total_bert_kg = 0.0
    total_rouge_kg = 0.0

    n = len(data)

    for entry in data:
        post = entry.get("post", "")
        gold = entry['implied_statement']
        baseline = entry['baseline_response']
        kg = entry['kg_response']

        bleu_baseline = sentence_bleu_score(gold, baseline)
        bert_baseline = sentence_similarity(gold, baseline)
        rouge_baseline = sentence_rouge_l(gold, baseline)

        bleu_kg = sentence_bleu_score(gold, kg)
        bert_kg = sentence_similarity(gold, kg)
        rouge_kg = sentence_rouge_l(gold, kg)

        total_bleu_baseline += bleu_baseline
        total_bert_baseline += bert_baseline
        total_rouge_baseline += rouge_baseline

        total_bleu_kg += bleu_kg
        total_bert_kg += bert_kg
        total_rouge_kg += rouge_kg

        evaluated_entry = {
            "post": post,
            "implied_statement": gold,
            "baseline_response": baseline,
            "kg_response": kg,
            "bleu_similarity_baseline": bleu_baseline,
            "bert_similarity_baseline": bert_baseline,
            "rouge_similarity_baseline": rouge_baseline,
            "bleu_similarity_kg": bleu_kg,
            "bert_similarity_kg": bert_kg,
            "rouge_similarity_kg": rouge_kg
        }

        evaluated_data.append(evaluated_entry)

    # Calculate averages
    averages = {
        "average_bleu_baseline": total_bleu_baseline / n if n else 0,
        "average_bert_baseline": total_bert_baseline / n if n else 0,
        "average_rouge_baseline": total_rouge_baseline / n if n else 0,
        "average_bleu_kg": total_bleu_kg / n if n else 0,
        "average_bert_kg": total_bert_kg / n if n else 0,
        "average_rouge_kg": total_rouge_kg / n if n else 0
    }

    # Save both detailed entries and averages in one JSON
    output = {
        "evaluations": evaluated_data,
        "averages": averages
    }

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(output, f_out, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {output_file}")


if __name__ == "__main__":
    input_files = [
        "results/prediction_results_llama3.1_prompt3_ITA.json",
        "results/prediction_results_mistral_7b_prompt3_ITA.json",
        "results/prediction_results_gemma2_9b_prompt3_ITA.json"
    ]

    for file in input_files:
        base, ext = os.path.splitext(file)
        evaluate_and_save(file, file)
