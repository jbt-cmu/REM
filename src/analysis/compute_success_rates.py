#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

# ========== CONFIGURATION PARAMETERS ==========
CSV_FILE = "src/trajectories/all_questions_aggregated_final.csv"  # Path to your aggregated CSV file.

# List of model answer columns to evaluate.
MODEL_LIST = [
    "answered_correctly_gpt-4o", 
    "answered_correctly_gemini-1.5-pro-latest", 
    "answered_correctly_gemini-1.5-flash-latest", 
    "answered_correctly_nova-lite-v1", 
    "answered_correctly_llama-3.2-11b-vision-instruct",
    "answered_correctly_gemini-1.5-flash-latest-TEXT",
    "answered_correctly_gemini-1.5-flash-latest-TEXT-NEW-PROMPT"
]

# Desired question types in preferred order.
DESIRED_QTYPES = [
    "comparison_in_frame",
    "comparison_out_of_frame",
    "left_right",
    "number",
    "order_preserving"
]

# Preset chance levels (in percent) for random guessing for each question type.
CHANCE_LEVELS = {
    "comparison_in_frame": 33.3,
    "comparison_out_of_frame": 33.3,
    "left_right": 50.0,
    "number": None,           # Undefined for "number" questions.
    "order_preserving": 33.3
}
# ==============================================

def compute_success_rate_all(df, model_col):
    """
    Compute the overall success rate for a given model column using all rows.
    A response is considered successful if its cleaned response (lowercased and stripped)
    contains the substring "yes"; otherwise, it is counted as wrong.
    Returns the success rate as a fraction (0 to 1).
    """
    responses = df[model_col].astype(str).str.strip().str.lower()
    successes = responses.str.contains("yes").astype(int)
    if len(successes) == 0:
        return np.nan
    return successes.mean()

def compute_success_rate_filtered(df, model_col):
    """
    Compute the overall success rate for a given model column using only rows where the
    response is unambiguous—i.e. exactly "yes" or "no". (Rows with any other response, such
    as "human_review", are thrown out.)
    Returns the success rate as a fraction (0 to 1).
    """
    filtered = df[df[model_col].astype(str).str.strip().str.lower().isin(["yes", "no"])]
    responses = filtered[model_col].astype(str).str.strip().str.lower()
    successes = responses.str.contains("yes").astype(int)
    if len(successes) == 0:
        return np.nan
    return successes.mean()

def normalize_table(results_df):
    """
    Given a results DataFrame (with models as rows and question types as columns) that includes
    a row named "Random Guess" containing the chance level for each question type,
    produce a normalized table.
    
    For each cell, apply:
         normalized = ((model_value - chance) / (100 - chance)) * 100.
    If chance is NaN or >= 100, the normalized value is set to NaN.
    Finally, the "Random Guess" row is replaced with zeros.
    """
    norm_df = results_df.drop(index="Random Guess").copy()
    chance = results_df.loc["Random Guess"]
    def normalize_value(val, ch):
        if pd.isna(ch) or pd.isna(val) or ch >= 100:
            return np.nan
        return ((val - ch) / (100 - ch)) * 100
    for col in norm_df.columns:
        norm_df[col] = norm_df[col].apply(lambda x: normalize_value(x, chance[col]))
    norm_df.loc["Random Guess"] = [0 if pd.notna(chance[col]) else np.nan for col in norm_df.columns]
    return norm_df

def main():
    parser = argparse.ArgumentParser(
        description="Compute overall success rates by question type for each model.\n"
                    "Table 1: Using all rows (human_review counted as wrong).\n"
                    "Table 2: Using only rows with unambiguous responses (human_review thrown out).\n\n"
                    "Then, produce normalized versions of these tables by subtracting the chance level\n"
                    "and scaling to the range 0-100 (chance becomes 0, perfect performance becomes 100)."
    )
    parser.add_argument("--csv", default=CSV_FILE,
                        help="Path to the aggregated CSV file (default: %(default)s)")
    args = parser.parse_args()
    
    # Read the CSV file.
    df = pd.read_csv(args.csv)
    
    # Determine the question types to process.
    question_types = [qt for qt in DESIRED_QTYPES if qt in df["question_type"].unique()]
    
    results_all = {}       # Table 1: using all rows.
    results_filtered = {}  # Table 2: using only unambiguous rows.
    
    # Loop over each model.
    for model in MODEL_LIST:
        model_all = {}
        model_filtered = {}
        for qt in question_types:
            df_qt = df[df["question_type"] == qt].copy()
            # For Table 1: use all rows (no filtering on response content).
            rate_all = compute_success_rate_all(df_qt, model)
            model_all[qt] = rate_all * 100 if not np.isnan(rate_all) else np.nan
            # For Table 2: only use rows where the response is exactly "yes" or "no".
            rate_filtered = compute_success_rate_filtered(df_qt, model)
            model_filtered[qt] = rate_filtered * 100 if not np.isnan(rate_filtered) else np.nan
        results_all[model] = model_all
        results_filtered[model] = model_filtered
    
    # Convert dictionaries to DataFrames.
    df_all = pd.DataFrame(results_all).T.round(1)
    df_filtered = pd.DataFrame(results_filtered).T.round(1)
    
    # Append a "Random Guess" row with preset chance levels.
    chance_series = pd.Series({qt: CHANCE_LEVELS.get(qt, np.nan) for qt in question_types})
    df_all.loc["Random Guess"] = chance_series
    df_filtered.loc["Random Guess"] = chance_series
    
    # Normalize the tables.
    norm_all = normalize_table(df_all)
    norm_filtered = normalize_table(df_filtered)
    
    # Print the results.
    print("Table 1: Overall Success Rate by Question Type (human_review counted as wrong):\n")
    print(df_all.to_string())
    print("\nTable 2: Overall Success Rate by Question Type (human_review thrown out):\n")
    print(df_filtered.to_string())
    print("\nNormalized Success Rates (using all rows) (chance subtracted and scaled 0-100):\n")
    print(norm_all.to_string())
    print("\nNormalized Success Rates (using filtered rows) (chance subtracted and scaled 0-100):\n")
    print(norm_filtered.to_string())

if __name__ == "__main__":
    main()

