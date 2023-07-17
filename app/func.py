import pandas as pd


def get_ontology(idustry_code=None):

    ontology = pd.read_csv("app/data/ontology.csv", index_col=0)
    ontology = ontology[ontology["code"] == idustry_code] if idustry_code else ontology
    print(type(ontology))
    return eval(ontology.to_json())