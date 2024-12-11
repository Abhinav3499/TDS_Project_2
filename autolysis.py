# Import necessary libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import argparse

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

def query_llm(messages, temperature=0.7, max_tokens=500):
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

def analyze_dataset(csv_filename):
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
        "sample_data": df.head(5).to_dict()
    }

    # Filter numerical columns
    numerical_cols = df.select_dtypes(include="number").columns
    if len(numerical_cols) > 0:
        # Generate correlation matrix for numerical columns
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig("correlation_matrix.png")
        plt.close()

        # Generate a distribution plot for the first numerical column
        sns.histplot(df[numerical_cols[0]].dropna(), kde=True, color="blue")
        plt.title(f"Distribution of {numerical_cols[0]}")
        plt.savefig("distribution_plot.png")
        plt.close()
    else:
        print("No numerical columns available for correlation or distribution analysis.")

    return summary

def generate_readme(data_summary, analysis_narrative):
    """
    Generate README.md with analysis narrative and references to charts.
    """
    with open("README.md", "w") as f:
        f.write("# Automated Dataset Analysis\n\n")
        f.write("## Dataset Summary\n")
        f.write(f"- Number of Rows: {data_summary['num_rows']}\n")
        f.write(f"- Number of Columns: {data_summary['num_columns']}\n")
        f.write("### Columns and Data Types:\n")
        for col, dtype in data_summary["columns"].items():
            f.write(f"- {col}: {dtype}\n")
        f.write("\n## Analysis Narrative\n")
        f.write(analysis_narrative)
        f.write("\n## Visualizations\n")
        f.write("1. Correlation Matrix:\n")
        f.write("![Correlation Matrix](correlation_matrix.png)\n")
        f.write("2. Distribution Plot:\n")
        f.write("![Distribution Plot](distribution_plot.png)\n")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Automated Dataset Analysis")
    parser.add_argument("csv_filename", help="Path to the CSV file to analyze")
    args = parser.parse_args()

    # Step 1: Analyze the dataset
    print("Analyzing dataset...")
    data_summary = analyze_dataset(args.csv_filename)

    # Step 2: Query LLM for narrative
    print("Generating narrative using LLM...")
    llm_messages = [
        {"role": "system", "content": "You are a data analyst."},
        {"role": "user", "content": f"Here is a summary of the dataset: {data_summary}. Provide an analysis and insights."}
    ]
    analysis_narrative = query_llm(llm_messages)

    # Step 3: Generate README.md
    print("Creating README.md...")
    generate_readme(data_summary, analysis_narrative)

    print("Analysis complete. Check README.md and the generated charts.")

if __name__ == "__main__":
    main()
