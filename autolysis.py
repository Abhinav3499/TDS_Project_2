import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
import requests
import argparse

# Global Constants
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

def query_llm(messages, temperature=0.7, max_tokens=500):
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(AIPROXY_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"LLM Query Failed: {response.status_code}\n{response.text}")

def load_dataset(csv_filename):
    try:
        return pd.read_csv(csv_filename, encoding="ISO-8859-1")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {e}")

def generate_visualizations(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    visuals = {}

    # Correlation Heatmap (if numerical data exists)
    numerical_cols = df.select_dtypes(include="number").columns
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        plt.figure(figsize=(6, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        filepath = os.path.join(output_dir, "correlation_matrix.png")
        plt.savefig(filepath, dpi=100)
        plt.close()
        visuals["Correlation Matrix"] = filepath

    # Distribution Plot
    if len(numerical_cols) > 0:
        col = numerical_cols[0]
        sns.histplot(df[col].dropna(), kde=True, color="blue")
        plt.title(f"Distribution of {col}")
        filepath = os.path.join(output_dir, "distribution_plot.png")
        plt.savefig(filepath, dpi=100)
        plt.close()
        visuals["Distribution Plot"] = filepath

    # Missing Values Heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    filepath = os.path.join(output_dir, "missing_values_heatmap.png")
    plt.savefig(filepath, dpi=100)
    plt.close()
    visuals["Missing Values Heatmap"] = filepath

    return visuals

def analyze_dataset(df):
    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
    }

    # Numerical Metrics
    numerical_cols = df.select_dtypes(include="number").columns
    if len(numerical_cols) > 0:
        summary["numerical_metrics"] = {
            col: {
                "mean": df[col].mean(),
                "std_dev": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "skewness": skew(df[col].dropna()),
                "kurtosis": kurtosis(df[col].dropna())
            }
            for col in numerical_cols
        }
    return summary

def generate_readme(data_summary, visuals, llm_narrative, output_dir):
    readme_content = f"""# Automated Dataset Analysis

## Dataset Summary
- Number of Rows: {data_summary['num_rows']}
- Number of Columns: {data_summary['num_columns']}
- Missing Values: {data_summary['missing_values']}
- Duplicate Rows: {data_summary['duplicates']}

## Insights and Analysis
{llm_narrative}

## Visualizations
"""
    for title, filepath in visuals.items():
        readme_content += f"### {title}\n![{title}]({filepath})\n"

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)

def main():
    parser = argparse.ArgumentParser(description="Automated Dataset Analysis")
    parser.add_argument("csv_filename", help="Path to the CSV file")
    args = parser.parse_args()

    output_dir = os.path.splitext(os.path.basename(args.csv_filename))[0]
    df = load_dataset(args.csv_filename)

    print("Generating visualizations...")
    visuals = generate_visualizations(df, output_dir)

    print("Analyzing dataset...")
    summary = analyze_dataset(df)

    print("Querying LLM for insights...")
    messages = [
        {"role": "system", "content": "You are a data analyst."},
        {"role": "user", "content": f"Here is a summary of the dataset: {summary}. Provide actionable insights and recommendations."}
    ]
    narrative = query_llm(messages)

    print("Creating README.md...")
    generate_readme(summary, visuals, narrative, output_dir)
    print(f"Analysis complete. Check output in {output_dir}/")

if __name__ == "__main__":
    main()
