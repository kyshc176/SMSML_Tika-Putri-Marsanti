import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Set alamat Tracking ke MLflow lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Lung Cancer Prediction")

# 2. Aktifkan autolog
mlflow.sklearn.autolog()

# 3. Load dataset
df = pd.read_csv('namadataset_preprocessing/survey_lung_cancer_processed.csv')

# 4. Pisahkan fitur dan target
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Mulai tracking MLflow
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Akurasi: {acc}")
