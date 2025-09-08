import base64
import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
from scipy.stats import zscore
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from pathlib import Path
from starlette.middleware.cors import CORSMiddleware

MODEL_DIR = Path("./models")
P_DEL_DIR = Path("./PaDEL-Descriptor")
P_DEL_JAR = P_DEL_DIR / "PaDEL-Descriptor.jar"
P_DEL_FINGERPRINT_XML = P_DEL_DIR / "PubchemFingerprinter.xml"

app = FastAPI(
    title="DiaPred: GLP-1 Receptor Agonist Prediction API",
    description="A deep learning-based API for predicting the activity of molecules against the GLP-1 Receptor. It utilizes a CNN model and PubChem fingerprints to classify compounds as 'Active' or 'Inactive' and provides a confidence score and applicability domain assessment."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SmilesList(BaseModel):
    smiles: List[str]

class PredictionResponse(BaseModel):
    smiles: str
    prediction: str
    confidence_score: float
    applicability_domain: str

cnn_model = None
pca_model = None

@app.on_event("startup")
async def startup_event():
    """Load the pre-trained deep learning and PCA models."""
    global cnn_model, pca_model
    try:
        cnn_model_path = MODEL_DIR / "CNN_Classifier.h5"
        pca_model_path = MODEL_DIR / "pca_model.joblib"
        
        cnn_model = tf.keras.models.load_model(cnn_model_path)
        pca_model = load(pca_model_path)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

def run_padel_descriptor(smiles_data: List[str], output_file_path: str) -> None:
    """Writes SMILES to a temporary file and runs PaDEL-Descriptor."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.smi') as temp_smi_file:
        temp_smi_file.write('\n'.join(smiles_data))
        temp_smi_file_path = temp_smi_file.name

    padel_jar_path = os.path.abspath(P_DEL_JAR)
    pubchem_fingerprinter_xml_path = os.path.abspath(P_DEL_FINGERPRINT_XML)
    temp_dir = os.path.abspath(Path(temp_smi_file_path).parent)

    padel_command = [
        "java", "-Xms256m", "-Xmx256m", "-Djava.awt.headless=true", "-jar", padel_jar_path,
        "-removesalt", "-standardizenitro", "-fingerprints",
        "-descriptortypes", pubchem_fingerprinter_xml_path,
        "-dir", temp_dir,
        "-file", output_file_path
    ]

    try:
        process = subprocess.run(padel_command, check=True, capture_output=True, text=True)
        os.remove(temp_smi_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"PaDEL-Descriptor.jar not found at {padel_jar_path}. Please ensure the directory and file exist.")
    except subprocess.CalledProcessError as e:
        print(f"PaDEL-Descriptor failed: {e.stderr}")
        raise RuntimeError("PaDEL-Descriptor failed to compute fingerprints. Check server logs for details.")

def calculate_applicability_domain(input_data: pd.DataFrame) -> np.ndarray:
    """Calculates applicability domain based on leverage threshold."""
    leverage_threshold = 0.11
    z_scores = zscore(input_data, axis=0)
    leverage = np.sum(z_scores ** 2, axis=1) / input_data.shape[1]
    return np.where(leverage > leverage_threshold, 'Outside', 'Inside')

@app.post("/predict/", response_model=List[PredictionResponse], summary="Predict GLP-1R agonist activity from SMILES strings")
async def predict_agonist(smiles_list: SmilesList):
    """
    Accepts a list of SMILES strings and returns predictions for each compound.
    """
    output_file_path = os.path.join(tempfile.gettempdir(), "descriptors_output.csv")
    
    run_padel_descriptor(smiles_list.smiles, output_file_path)

    desc = pd.read_csv(output_file_path).iloc[:, 1:]
    
    if desc.shape[1] != 881:
        raise ValueError(f"Expected 881 features from PaDEL, but got {desc.shape[1]}.")

    desc.columns = desc.columns.astype(str)
    desc.reset_index(drop=True, inplace=True)

    desc_pca = pca_model.transform(desc)
    
    desc_reshaped = np.reshape(desc_pca, (desc_pca.shape[0], desc_pca.shape[1], 1))

    predictions = cnn_model.predict(desc_reshaped).flatten()
    applicability_domain = calculate_applicability_domain(desc_pca)
    
    response_data = []
    for i, smiles_string in enumerate(smiles_list.smiles):
        prediction = 'Active' if predictions[i] > 0.5 else 'Inactive'
        response_data.append(PredictionResponse(
            smiles=smiles_string,
            prediction=prediction,
            confidence_score=float(predictions[i]),
            applicability_domain=applicability_domain[i]
        ))

    os.remove(output_file_path)
    
    return response_data

@app.post("/predict_file/", response_model=List[PredictionResponse], summary="Predict GLP-1R agonist activity from a SMILES file")
async def predict_agonist_file(file: UploadFile = File(...)):
    """
    Accepts a file containing SMILES strings (one per line) and returns predictions.
    """
    smiles_content = await file.read()
    smiles_list_str = smiles_content.decode("utf-8").splitlines()

    return await predict_agonist(SmilesList(smiles=smiles_list_str))
