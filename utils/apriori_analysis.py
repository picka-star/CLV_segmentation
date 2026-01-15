import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def run_apriori(df, min_support=0.02, min_confidence=0.3):
    basket = pd.crosstab(df["Transaction_ID"], df["Product_Category"])
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    freq_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)

    return rules
