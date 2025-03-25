from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from ..config import MODEL_PATH, PARAM_GRID, RANDOM_STATE, TEST_SIZE

def train_model(X, y, date_series):
    """Train and evaluate the model with time-based split"""
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
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluation
    y_pred = best_model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return best_model