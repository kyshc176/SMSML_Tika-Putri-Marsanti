import requests
import json
import time
import logging
import random

# Konfigurasi logging (opsional, tapi bagus untuk ada)
logging.basicConfig(filename="api_model_logs.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Endpoint API yang benar menunjuk ke exporter di port 8001
API_URL = "http://127.0.0.1:8001/predict"

# Input data dasar
base_input_data = {
    "dataframe_split": {
        "columns": [
            "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
            "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE", "ALLERGY",
            "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
            "SWALLOWING DIFFICULTY", "CHEST PAIN"
        ],
        "data": [
            [0, 69, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2]
        ]
    }
}

headers = {"Content-Type": "application/json"}

print("Mulai mengirim request prediksi secara berkala... Tekan Ctrl+C untuk berhenti.")

# === PERBAIKAN UTAMA: MENAMBAHKAN LOOP ===
while True:
    try:
        # Menggunakan requests dengan argumen 'json' lebih aman
        response = requests.post(API_URL, json=base_input_data)

        if response.status_code == 200:
            prediction = response.json()
            print(f"Prediksi Sukses: {prediction}")
            logging.info(f"Response: {prediction}")
        else:
            # Jika ada error, cetak isi dari error tersebut
            print(f"Error {response.status_code}: {response.text}")
            logging.error(f"Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"Koneksi ke {API_URL} gagal. Pastikan container exporter berjalan.")
        break
    except KeyboardInterrupt:
        print("\nScript dihentikan.")
        break
    except Exception as e:
        print(f"Terjadi error tak terduga: {e}")
        logging.error(f"Exception: {str(e)}")
        break
    
    # Tunggu 3 detik sebelum mengirim request berikutnya
    time.sleep(3)

