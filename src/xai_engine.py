import pandas as pd
import numpy as np
import xgboost as xgb
import shap

class QATriageEngine:
    def __init__(self):
        # We use a classifier to predict if a tile "Needs Review" (1) or is "Good" (0)
        self.model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            random_state=42,
            min_child_weight=0.001  # Allows splitting on tiny data
        )
        self.explainer = None
        self.feature_cols = ['mean_brightness', 'image_contrast', 'edge_complexity']

    def train_triage_model(self, df: pd.DataFrame):
        """
        Trains the XGBoost model to predict QA failure based on image features.
        """
        # Define our target: Let's say any tile with an IoU < 0.75 needs human review
        df['needs_review'] = (df['iou_score'] < 0.75).astype(int)
        
        X = df[self.feature_cols]
        y = df['needs_review']
        
        # Train the model
        self.model.fit(X, y)
        
        # Initialize the SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        print("Triage model trained and SHAP explainer initialized.")

    def explain_tile_failure(self, tile_features: pd.DataFrame) -> dict:
        """
        Generates a human/LLM-readable explanation for WHY a specific tile was flagged.
        Returns a dictionary of feature contributions.
        """
        if self.explainer is None:
            raise ValueError("Model must be trained before generating explanations.")

        # Ensure we only pass the feature columns
        X_tile = tile_features[self.feature_cols]
        
        # Predict probability of needing review
        risk_score = self.model.predict_proba(X_tile)[0][1]
        
        # Calculate SHAP values for this specific tile
        shap_values = self.explainer.shap_values(X_tile)
        
        # Extract the base value (average risk across all tiles)
        base_value = self.explainer.expected_value
        
        # Map features to their SHAP impact scores
        # Positive SHAP value = increases risk of failure
        # Negative SHAP value = decreases risk of failure
        feature_impacts = {}
        for i, col in enumerate(self.feature_cols):
            feature_impacts[col] = round(float(shap_values[0][i]), 4)
            
        # Sort by absolute impact to find the "primary reasons"
        sorted_impacts = dict(sorted(feature_impacts.items(), key=lambda item: abs(item[1]), reverse=True))

        return {
            "risk_of_failure_percent": round(risk_score * 100, 2),
            "base_risk_percent": round(1 / (1 + np.exp(-base_value)) * 100, 2), # Convert log-odds to prob
            "top_contributing_factors": sorted_impacts
        }

# --- Test the function with Mock Data ---
if __name__ == "__main__":
    # 1. Create mock merged data (simulating the output of Steps 1 & 2)
    mock_data = pd.DataFrame({
        'tile_id':['tile_001', 'tile_002', 'tile_003', 'tile_004', 'tile_005'],
        'iou_score':[0.85, 0.45, 0.90, 0.30, 0.80],
        'mean_brightness':[140, 80, 150, 75, 135], # 80 and 75 are very dark
        'image_contrast':[45, 15, 50, 10, 40],     # 15 and 10 are washed out
        'edge_complexity':[1200, 3500, 1100, 4000, 1300] # 3500+ is dense vegetation
    })

    # 2. Initialize and train the engine
    engine = QATriageEngine()
    engine.train_triage_model(mock_data)

    # 3. Explain a failing tile (e.g., tile_002)
    bad_tile = mock_data[mock_data['tile_id'] == 'tile_002']
    explanation = engine.explain_tile_failure(bad_tile)
    
    print(f"\nExplanation for Tile 002:")
    print(explanation)