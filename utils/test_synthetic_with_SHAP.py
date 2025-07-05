import pandas as pd
import numpy as np
import shap
import joblib
import yaml
from pathlib import Path
from foetal_health_predictor import FoetalHealthModel
from utils.synthetic_data_generator import generate_synthetic_data

# === Constants ===
N_SYNTHETIC_SAMPLES = 100

# === Project Setup ===
project_root = Path(__file__).resolve().parents[1]
test_data_path = project_root / "data" / "processed" / "test.csv"
artifacts_dir = project_root / "train_eval_scripts" / "artifacts"
config_path = project_root / "configs" / "selected_columns.yaml"

def generate_prediction_insights():
    # Load test data and config
    df_test = pd.read_csv(test_data_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    selected_features = config["selected_columns"]

    # Generate and prepare synthetic data
    synthetic_df = generate_synthetic_data(df_test, n_samples=N_SYNTHETIC_SAMPLES)
    synthetic_df_selected = synthetic_df[selected_features]
    X_test = synthetic_df_selected.drop("fetal_health", axis=1)
    y_test = synthetic_df_selected["fetal_health"]

    # Load model
    model = joblib.load(artifacts_dir / "best_random_forest.pkl")
    model_wrapper = FoetalHealthModel()
    model_wrapper.model = model

    # SHAP Explanation for a single sample
    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(n=1, random_state=np.random.randint(0, N_SYNTHETIC_SAMPLES))
    shap_values = explainer.shap_values(sample)

    # Prediction and probabilities
    predicted_label = model.predict(sample)[0]
    label_mapping = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    label_name = label_mapping.get(predicted_label, "Unknown")
    probs = model.predict_proba(sample)[0]
    predicted_class = np.where(model.classes_ == predicted_label)[0][0]
    predicted_probability = probs[predicted_class]

    # Prediction Insights
    pred_class_shap = shap_values[0, :, predicted_class]
    feature_shap_pairs = list(zip(sample.columns, pred_class_shap))
    sorted_features = sorted(feature_shap_pairs, key=lambda x: abs(x[1]), reverse=True)
    top_features = [f for f, _ in sorted_features[:3]]
    top_shap_values = [v for _, v in sorted_features[:3]]

    return {
        "predicted_label": label_name,
        "predicted_probability": predicted_probability,
        "top_features": top_features,
        "top_shap_values": top_shap_values
    }

if __name__ == "__main__":
    synthetic_prediction_insights = generate_prediction_insights()
    print(synthetic_prediction_insights)
