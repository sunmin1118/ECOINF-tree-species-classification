#Copyright (c) [2025] Min Sun. All rights reserved.
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

data = pd.read_excel("")
results_df = pd.DataFrame(columns=['Classifier', 'McNemar', 'p', 'significance', 'comparison', 'B', 'C'])
for base_clf in ['MLP', 'RF', 'SVM', 'CNN']:
    table = pd.crosstab(data[base_clf], data['Ensemble'])
    result = mcnemar(table, exact=False, correction=False)
    b = table.iloc[1, 0]
    c = table.iloc[0, 1]
    significance = "p < 0.05" if result.pvalue < 0.05 else "p ≥ 0.05"

    comparison = ""
    if result.pvalue < 0.05:
        if c > b:
            comparison = "ensemble model outperformed" + base_clf
        else:
            comparison = base_clf + "significant outperformed"
    results_df = pd.concat([results_df, pd.DataFrame({
        'Classifier': [f"{base_clf} vs Ensemble"],
        'McNemar': [f"{result.statistic:.3f}"],
        'p': [f"{result.pvalue:.4f}"],
        'significance': [significance],
        'comparison': [comparison],
        'B': [b],
        'C': [c]
    })], ignore_index=True)
    print(f"\n{base_clf} vs Ensemble:")
    print(f"table:\n{table}")
    print(f"B: {b}")
    print(f"C: {c}")
    print(f"McNemar: {result.statistic:.3f}")
    print(f"p: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("p < 0.05")
        if c > b:
            print("ensemble model outperformed" + base_clf)
        else:
            print(base_clf + "outperformed ensemble model")
    else:
        print("p ≥ 0.05")
output_path = ""
results_df.to_excel(output_path, index=False)
print(f"\nresults: {output_path}")
