import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import dagshub
import shutil
import time

# Inisialisasi dagshub
dagshub.init(repo_owner='super-nayr', repo_name='lungcancer-mlflow', mlflow=True)

mlflow.set_experiment("Lung Cancer Prediction")

df = pd.read_csv('namadataset_preprocessing/survey_lung_cancer_processed.csv')

X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for n_estimators in [50, 100]:
    for max_depth in [5, 10]:
        run_name = f"RF_n{n_estimators}_d{max_depth}"
        with mlflow.start_run(run_name=run_name) as run:
            start_time = time.time()

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", 42)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            train_duration = time.time() - start_time
            mlflow.log_metric("train_duration_seconds", train_duration)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            prec_class_0 = class_report.get('0', {}).get('precision', 0)
            prec_class_1 = class_report.get('1', {}).get('precision', 0)
            rec_class_0 = class_report.get('0', {}).get('recall', 0)
            rec_class_1 = class_report.get('1', {}).get('recall', 0)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", prec)
            mlflow.log_metric("recall_macro", rec)
            mlflow.log_metric("f1_score_macro", f1)
            mlflow.log_metric("precision_class_0", prec_class_0)
            mlflow.log_metric("precision_class_1", prec_class_1)
            mlflow.log_metric("recall_class_0", rec_class_0)
            mlflow.log_metric("recall_class_1", rec_class_1)

            input_example = X_test.iloc[:1]
            signature = infer_signature(X_test, y_pred)

            mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=signature)

            # Buat confusion matrix plot dan simpan sementara, upload, lalu hapus file
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            cm_path = "training_confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)

            # Simpan metrics ke JSON, upload, lalu hapus
            metric_info = {
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_score_macro": f1,
                "precision_class_0": prec_class_0,
                "precision_class_1": prec_class_1,
                "recall_class_0": rec_class_0,
                "recall_class_1": rec_class_1,
                "train_duration_seconds": train_duration
            }
            metric_path = "metric_info.json"
            with open(metric_path, "w") as f:
                json.dump(metric_info, f, indent=4)
            mlflow.log_artifact(metric_path)
            os.remove(metric_path)

            # Simpan estimator.html, upload, lalu hapus
            estimator_html_path = "estimator.html"
            with open(estimator_html_path, "w") as f:
                f.write(f"""
                <html>
                    <body>
                        <h2>Random Forest Model</h2>
                        <ul>
                            <li>n_estimators: {n_estimators}</li>
                            <li>max_depth: {max_depth}</li>
                            <li>random_state: 42</li>
                            <li>train_duration_seconds: {train_duration:.2f}</li>
                        </ul>
                    </body>
                </html>
                """)
            mlflow.log_artifact(estimator_html_path)
            os.remove(estimator_html_path)
