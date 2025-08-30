# Anomaly-detection-with-VLM-description
Code used in Master Thesis "Detekcija anomalija u industrijskom okruženju"


## How to run

### PolanecAD dataset

Download both PolanecAD and PolanecVLM datasets from Hugging Face: https://huggingface.co/datasets/tonipol/PolanecAD and https://huggingface.co/datasets/tonipol/PolanecVLM

PolanecAD is for anomaly detection model, PolanecVLM is for VLM description model.


### Anomaly detection model

1. Go into `AnomalyDetectionModel` directory
2. Setup virtual environment and install required dependencies
3. Download pretrained weights for SimpleNet model from Hugging Face: https://huggingface.co/tonipol/simplenet-polanecad and put them in `AnomalyDetectionModel\results\PolanecAD_Results\simplenet_polanecad\run_40ep\models\0\polanecad_tablet` directory
4. Put PolanecAD dataset in `AnomalyDetectionModel\PolanecAD` directory
5. Run the model with `run.sh` script

### VLM description model
1. Go into `VLMDescriptionModel` directory
2. Download pretrained weights for VLM model from Hugging Face: https://huggingface.co/tonipol/blip2-opt-2.7b-anomaly-detection-description (only if running locally)
3. Run `VLM-AD-description.ipynb` notebook
