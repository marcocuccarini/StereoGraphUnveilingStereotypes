import pandas as pd

# File paths
file_paths = [
    'manual_evaluation_gemma2 - prediction_results_gemma2_9b_prompt3_ITA_clean_scored_sorted_random100.csv',
    'manual_evaluation_llama3.1 - prediction_results_llama3.csv',
    'manual_evaluation_mistral - prediction_results_mistral_7b_prompt3_ITA_clean_scored_sorted_random100.csv'
]

# Evaluation column mappings
participants_baseline = {
    'Marco': 'BASE_EVAL_Marco',
    'Lia': 'BASE_EVAL_Lia',
    'Bea': 'BASE_EVAL_Bea'
}

participants_kg = {
    'Marco': 'KG_EVAL_Marco',
    'Lia': 'KG_EVAL_Lia',
    'Bea': 'KG_EVAL_bea'
}

participants_stereo = {
    'Marco': 'Stereo_eval_Marco',
    'Lia': 'Stereo_eval_Lia',
    'Bea': 'Stereo_eval_bea'
}

# Normalization parameters
score_min, score_max = 1, 5

# Process each file
for file in file_paths:
    print(f"\n📄 Processing file: {file}")
    try:
        df = pd.read_csv(file, sep=";")

        print("🔹 Normalized Baseline Averages (0–1):")
        for person, col in participants_baseline.items():
            scores = pd.to_numeric(df[col], errors='coerce')
            avg = scores.dropna().mean()
            norm = (avg - score_min) / (score_max - score_min) if avg is not None else None
            print(f"  {person}: {norm:.3f}")

        print("🔸 Normalized KG Averages (0–1):")
        for person, col in participants_kg.items():
            scores = pd.to_numeric(df[col], errors='coerce')
            avg = scores.dropna().mean()
            norm = (avg - score_min) / (score_max - score_min) if avg is not None else None
            print(f"  {person}: {norm:.3f}")

        print("🔻 Normalized Stereotype Percentages (0–1):")
        for person, col in participants_stereo.items():
            total = df[col].notna().sum()
            si_count = df[col].str.upper().eq('SI').sum()
            percentage = (si_count / total) if total > 0 else 0
            print(f"  {person}: {percentage:.3f}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
