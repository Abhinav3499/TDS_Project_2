# Import necessary libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import argparse
import numpy as np
from scipy.stats import skew, kurtosis

# Ensure environment variable for AI Proxy Token is set
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    exit(1)

# Headers for AI Proxy requests
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

def query_llm(messages, temperature=0.7, max_tokens=1000):
    """
    Query the LLM via AI Proxy.
    """
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
        print(f"Error querying LLM: {response.status_code}\n{response.text}")
        exit(1)

def analyze_dataset(csv_filename, output_dir):
    """
    Perform basic analysis on the dataset and return a summary.
    """
    try:
        # Load the dataset
        df = pd.read_csv(csv_filename, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

    # Basic information about the dataset
    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head(5).to_dict(),
        "duplicates": df.duplicated().sum()
    }

    # Detect numerical columns
    numerical_cols = df.select_dtypes(include="number").columns
    if len(numerical_cols) > 0:
        # Advanced metrics
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

        # Detect outliers using Z-score
        z_scores = np.abs((df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std())
        summary["outliers"] = {
            col: (z_scores[col] > 3).sum() for col in numerical_cols
        }

        # Generate correlation matrix
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()

        # Generate a distribution plot for all numerical columns
        for col in numerical_cols:
            sns.histplot(df[col].dropna(), kde=True, color="blue")
            plt.title(f"Distribution of {col}")
            plt.savefig(os.path.join(output_dir, f"distribution_{col}.png"))
            plt.close()

        # Generate a pair plot for numerical columns
        sns.pairplot(df[numerical_cols])
        plt.suptitle("Pair Plot for Numerical Columns", y=1.02)
        plt.savefig(os.path.join(output_dir, "pair_plot.png"))
        plt.close()
    else:
        print("No numerical columns available for analysis.")

    # Detect categorical columns
    categorical_cols = df.select_dtypes(include="object").columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            sns.countplot(y=df[col].dropna(), order=df[col].value_counts().index)
            plt.title(f"Bar Chart for {col}")
            plt.savefig(os.path.join(output_dir, f"bar_chart_{col}.png"))
            plt.close()

    # Missing data heatmap
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.savefig(os.path.join(output_dir, "missing_values_heatmap.png"))
    plt.close()

    return summary

def generate_readme(data_summary, analysis_narrative, output_dir):
    """
    Generate README.md with analysis narrative and references to charts.
    """
    visualizations = "\n".join([
        f"{i+1}. ![{viz}](./{viz})" for i, viz in enumerate(os.listdir(output_dir)) if viz.endswith('.png')
    ])

    readme_content = f"""# Automated Dataset Analysis

## Dataset Summary
- Number of Rows: {data_summary['num_rows']}
- Number of Columns: {data_summary['num_columns']}
- Duplicate Rows: {data_summary['duplicates']}

### Columns and Data Types:
""" + "\n".join([f"- {col}: {dtype}" for col, dtype in data_summary["columns"].items()]) + f"""

### Numerical Metrics:
""" + "\n".join([
        f"- {col}: Mean={metrics['mean']}, Std Dev={metrics['std_dev']}, Min={metrics['min']}, Max={metrics['max']}, Skewness={metrics['skewness']}, Kurtosis={metrics['kurtosis']}"
        for col, metrics in data_summary.get("numerical_metrics", {}).items()
    ]) + f"""

## Analysis Narrative
{analysis_narrative}

## Visualizations
{visualizations}
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Automated Dataset Analysis")
    parser.add_argument("csv_filename", help="Path to the CSV file to analyze")
    args = parser.parse_args()

    # Create output directory
    dataset_name = os.path.splitext(os.path.basename(args.csv_filename))[0]
    output_dir = os.path.join(os.getcwd(), dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Analyze the dataset
    print("Analyzing dataset...")
    data_summary = analyze_dataset(args.csv_filename, output_dir)

    # Step 2: Query LLM for narrative
    print("Generating narrative using LLM...")
    llm_messages = [
        {"role": "system", "content": "You are a data analyst."},
        {"role": "user", "content": f"Here is a summary of the dataset: {data_summary}. Provide advanced insights, key findings, and potential improvements."}
    ]
    analysis_narrative = query_llm(llm_messages)

    # Step 3: Generate README.md
    print("Creating README.md...")
    generate_readme(data_summary, analysis_narrative, output_dir)

    print("Analysis complete. Check README.md and the generated charts.")

if __name__ == "__main__":
    main()
