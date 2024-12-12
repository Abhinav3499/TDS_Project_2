import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import argparse
import numpy as np
from scipy import stats

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
    Query the LLM via AI Proxy with enhanced context and token limit.
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
        return "Unable to generate narrative."

def advanced_data_analysis(df):
    """
    Perform advanced statistical analysis on the dataset.
    """
    analysis_results = {}
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        analysis_results['numerical_analysis'] = {}
        for col in numerical_cols:
            col_data = df[col].dropna()
            analysis_results['numerical_analysis'][col] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std_dev': col_data.std(),
                'skewness': stats.skew(col_data),
                'kurtosis': stats.kurtosis(col_data),
                'outliers': detect_outliers(col_data)
            }
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        analysis_results['categorical_analysis'] = {}
        for col in categorical_cols:
            analysis_results['categorical_analysis'][col] = {
                'unique_values': df[col].nunique(),
                'top_5_values': df[col].value_counts().head().to_dict()
            }
    
    return analysis_results

def detect_outliers(data, method='iqr'):
    """
    Detect outliers using IQR or Z-score method.
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
    else:  # Z-score method
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > 3]
    
    return {
        'count': len(outliers),
        'percentage': len(outliers) / len(data) * 100
    }

def create_comprehensive_visualizations(df, output_dir):
    """
    Create a comprehensive set of visualizations.
    """
    plt.figure(figsize=(15, 10))
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Correlation heatmap
    if len(numerical_cols) > 1:
        plt.subplot(2, 2, 1)
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
    
    # Distribution of first numerical column
    if len(numerical_cols) > 0:
        plt.subplot(2, 2, 2)
        sns.histplot(df[numerical_cols[0]], kde=True)
        plt.title(f'Distribution of {numerical_cols[0]}')
    
    # Categorical column analysis
    if len(categorical_cols) > 0:
        plt.subplot(2, 2, 3)
        df[categorical_cols[0]].value_counts().plot(kind='bar')
        plt.title(f'Top Values in {categorical_cols[0]}')
        plt.xticks(rotation=45)
    
    # Missing values heatmap
    plt.subplot(2, 2, 4)
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'))
    plt.close()

def generate_comprehensive_readme(df, analysis_results, output_dir):
    """
    Generate a comprehensive README with detailed insights.
    """
    readme_content = f"""# Comprehensive Dataset Analysis

## Dataset Overview
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}

## Columns Summary
{generate_columns_summary(df)}

## Numerical Analysis
{generate_numerical_analysis(analysis_results)}

## Categorical Analysis
{generate_categorical_analysis(analysis_results)}

## Visualizations
![Comprehensive Analysis](comprehensive_analysis.png)
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)

def generate_columns_summary(df):
    return "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])

def generate_numerical_analysis(analysis_results):
    if 'numerical_analysis' not in analysis_results:
        return "No numerical columns found."
    
    return "\n".join([
        f"### {col}\n"
        f"- Mean: {stats['mean']:.2f}\n"
        f"- Median: {stats['median']:.2f}\n"
        f"- Standard Deviation: {stats['std_dev']:.2f}\n"
        f"- Skewness: {stats['skewness']:.2f}\n"
        f"- Kurtosis: {stats['kurtosis']:.2f}\n"
        f"- Outliers: {stats['outliers']['count']} ({stats['outliers']['percentage']:.2f}%)"
        for col, stats in analysis_results['numerical_analysis'].items()
    ])

def generate_categorical_analysis(analysis_results):
    if 'categorical_analysis' not in analysis_results:
        return "No categorical columns found."
    
    return "\n".join([
        f"### {col}\n"
        f"- Unique Values: {stats['unique_values']}\n"
        f"- Top 5 Values: {stats['top_5_values']}"
        for col, stats in analysis_results['categorical_analysis'].items()
    ])

def main():
    parser = argparse.ArgumentParser(description="Advanced Dataset Analysis")
    parser.add_argument("csv_filename", help="Path to the CSV file to analyze")
    args = parser.parse_args()

    # Create output directory
    dataset_name = os.path.splitext(os.path.basename(args.csv_filename))[0] output_dir = os.path.join(os.getcwd(), dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load the dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv(args.csv_filename, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

    # Step 2: Perform advanced data analysis
    print("Performing advanced data analysis...")
    analysis_results = advanced_data_analysis(df)

    # Step 3: Create comprehensive visualizations
    print("Creating visualizations...")
    create_comprehensive_visualizations(df, output_dir)

    # Step 4: Query LLM for narrative
    print("Generating narrative using LLM...")
    llm_messages = [
        {"role": "system", "content": "You are a data analyst."},
        {"role": "user", "content": f"Here is a summary of the dataset: {analysis_results}. Provide a detailed analysis and insights."}
    ]
    analysis_narrative = query_llm(llm_messages)

    # Step 5: Generate README.md
    print("Creating README.md...")
    generate_comprehensive_readme(df, analysis_results, output_dir)

    print("Analysis complete. Check README.md and the generated visualizations.")

if __name__ == "__main__":
    main()