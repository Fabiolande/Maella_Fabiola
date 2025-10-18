# src/train.py (version avec MLflow) 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline 
import mlflow 
import mlflow.sklearn 
import os 
from sklearn.metrics import accuracy_score
import joblib
 
# Définir le nom de l'expérience MLflow 
mlflow.set_experiment("Analyse de Sentiments Twitter") 
 
def train_model(model_name, pipeline): 
    """Entraîne un modèle et log les informations avec MLflow.""" 
    with mlflow.start_run(run_name=model_name): 
        # Charger les données 
        train_df = pd.read_csv(os.path.join('data', 'train.csv')) 
        X_train = train_df['text'].astype(str) 
        y_train = train_df['sentiment'] 

        # Logger les paramètres du modèle 
        params = pipeline.get_params() 
        mlflow.log_param("model_type", model_name) 
        mlflow.log_param("tfidf_ngram_range", params['tfidf__ngram_range']) 
        mlflow.log_param("tfidf_max_features", params['tfidf__max_features']) 
        if model_name == 'LogisticRegression': 
            mlflow.log_param("clf_max_iter", params['clf__max_iter']) 

        # Entraînement 
        print(f"Entraînement du modèle {model_name}...") 
        pipeline.fit(X_train, y_train) 
        print("Entraînement terminé.") 

        # Prédiction sur le train (ou validation si tu en as)
        y_pred = pipeline.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        mlflow.log_metric("train_accuracy", acc)

        # Logger le modèle comme un artefact 
        mlflow.sklearn.log_model(pipeline, f"{model_name}_pipeline")
        
        # Sauvegarde locale pour l'évaluation
        os.makedirs("models", exist_ok=True)
        if model_name == "LogisticRegression":
            joblib.dump(pipeline, os.path.join("models", "logistic_regression_pipeline.joblib"))
        elif model_name == "NaiveBayes":
            joblib.dump(pipeline, os.path.join("models", "naive_bayes_pipeline.joblib"))
            
        print(f"Modèle {model_name} et paramètres loggés avec MLflow.")  
if __name__ == "__main__": 
        # Pipeline pour la Régression Logistique 

        lr_pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]) 
        train_model('LogisticRegression', lr_pipeline) 
        # Pipeline pour Naive Bayes 
        nb_pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("mnb", MultinomialNB())]) 
        train_model('NaiveBayes', nb_pipeline)