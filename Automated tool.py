"""
Title: Automated tool for the detection of structural faults through machine learning models and seismic signal processing
Author: Anshel Chuquiviguel Zaña
Contact: anshel.chuquiviguel@utec.edu.pe
Created: February 19, 2025
Last Updated: February 19, 2025
Version: 1.0
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath('Sources'))
import about_ml_models as mlm

#falla =True -> para obtener datos con fallas estructurales
#falla =False -> para obtener datos sin fallas estructurales
#falla =None -> selección aleatoria de los datos

data_IM, X_new_scaled=mlm.get_seismic_signals(falla=None)

# Load models
models_det = mlm.load_models()

# Make predictions
results, consensus = mlm.predict_damage(X_new_scaled, models_det)

# damage location prediction
if consensus>0.5:
    mlm.ubicar_falla(data_IM)
