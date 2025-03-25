from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from config import MODEL_PATH, PARAM_GRID, RANDOM_STATE, TEST_SIZE

def train_model(X, y, date_series):
    """Train and evaluate the model with time-based split"""
    # First ensure y contains only integers (0 and 1)
    y = y.astype(int)
    
    # Verify we have binary classification targets
    unique_classes = set(y)
    if unique_classes != {0, 1}:
        raise ValueError(f"Target contains unexpected classes: {unique_classes}. Expected only 0 and 1.")
    
    # Time-based split
    split_date = date_series.quantile(0.8)
    X_train = X[date_series < split_date]
    X_test = X[date_series >= split_date]
    y_train = y[date_series < split_date]
    y_test = y[date_series >= split_date]
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    
    # Grid search
    grid_search = GridSearchCV(pipeline, PARAM_GRID, cv=5, n_jobs=-1, verbose=1)
    print("Training model with grid search...")
    grid_search.fit(X_train, y_train)
    
     # Save the best model
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return grid_search