from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def get_traditional_model(model_name):
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=15, random_state=42)
    else:
        raise ValueError(f"Unknown traditional model: {model_name}")