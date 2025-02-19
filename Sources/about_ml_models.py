import numpy as np
import pandas as pd
import seismic_signal_analysis as ssa
import matplotlib.pyplot as plt
from tabulate import tabulate
import joblib
import pickle
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

def aplicar_escalamiento(X, scalers_dict, im_names=None, n_pisos=5):
    """
    Aplica escalamiento previamente ajustado a nuevos datos.

    Parámetros:
    X : DataFrame - Nuevos datos a escalar
    scalers_dict : dict - Diccionario de scalers por IM
    im_names : list - Lista de nombres de IMs
    n_pisos : int - Número de pisos

    Retorna:
    DataFrame: Datos escalados
    """
    if im_names is None:
        im_names = ['PGA', 'PGV', 'IA', 'CAV', 'RMS', 'DS', 'FP', 'IH', 'ET', 'EWT']
    X_scaled = X.copy()

    for im in im_names:
        cols_im = [f"{im}_{i + 1}" for i in range(n_pisos)]
        X_scaled[cols_im] = scalers_dict[im].transform(X[cols_im])

    return X_scaled


# %%
def get_seismic_signals(falla=None):
    structures = {
        'E1': 4,
        'E2': 6,
        'E3': 8,
        'E4': 8
    }
    seismic_events = {
        'A': 'Northridge, 1/17/1994, Los Angeles, California',
        'B': 'Cape Mendocino, 4/25/1992, Northern California',
        'C': 'Imperial Valley, 10/15/1979, El Centro'
    }

    with open('DataSet/DataSet_seismic_signals.pkl', 'rb') as file:
        DataSet = pickle.load(file)

    if falla is True or falla is False:
        for ids in DataSet:
            signal_data = np.random.choice(DataSet)
            if signal_data['falla'] == falla: break
    else:
        signal_data = np.random.choice(DataSet)

    data = signal_data['Datos']
    name = signal_data['nombre']
    structure = signal_data['estructura']
    damage = signal_data['falla']
    earthquake = signal_data['sismo']
    floor = signal_data['piso']
    severity = signal_data['grado']

    # Plot configuration
    plt.figure(figsize=(12, 6))
    for i in range(1, 6):
        plt.plot(data[:, 0], data[:, i], label=f'Floor {i} Signal', linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.title(f"Seismic Response - {name}", pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print general information
    print("\n" + "=" * 80)
    print("SEISMIC SIGNAL REPORT")
    print("=" * 80 + "\n")

    print("GENERAL INFORMATION:")
    print("-" * 50)
    print(f"Signal ID: {name}")
    print(f"Structure {structure}: {structures[structure]} columns, 5 levels")
    print(f"Earthquake {earthquake}: {seismic_events[earthquake]}")
    print(f"Damage Status: {'Damaged' if damage else 'Undamaged'}")
    if damage:
        print(f"Damage Location: Floor {floor}")
        print(f"Damage Severity: Level {severity}")
    print("-" * 50 + "\n")

    # Intensity Measures configuration
    IM = ['PGA', 'PGV', 'Intensidad_Arias', 'CAV', 'RMS_Aceleracion',
          'Duracion_Significativa', 'Frecuencia_Predominante', 'Intensidad_Housner',
          'Energia_Tiempo', 'Energia_Wavelet_energia_total']
    nIM = ['PGA', 'PGV', 'IA', 'CAV', 'RMS', 'DS', 'FP', 'IH', 'ET', 'EWT']

    im_descriptions = {
        'PGA': 'Peak Ground Acceleration',
        'PGV': 'Peak Ground Velocity',
        'IA': 'Arias Intensity',
        'CAV': 'Cumulative Absolute Velocity',
        'RMS': 'Root Mean Square Acceleration',
        'DS': 'Significant Duration',
        'FP': 'Predominant Frequency',
        'IH': 'Housner Intensity',
        'ET': 'Time Energy',
        'EWT': 'Wavelet Energy Total'
    }

    # Calculate IMs
    intensity_measures = ssa.analizar_piso(data[1:, :])

    # Print Intensity Measures
    print("INTENSITY MEASURES BY FLOOR:")
    print("-" * 50)

    # Prepare data for tabulate
    table_data = []
    for i, (long_name, short_name) in enumerate(zip(IM, nIM)):
        row = [short_name, im_descriptions[short_name]]
        row.extend([f"{intensity_measures[long_name][j]:.4f}" for j in range(5)])
        table_data.append(row)

    headers = ["IM", "Description", "Floor 1", "Floor 2", "Floor 3", "Floor 4", "Floor 5"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\n")

    # Prepare data for scaling
    dict_im = {}
    for i in range(len(IM)):
        for j in range(1, 6):
            dict_im[f'{nIM[i]}_{j}'] = intensity_measures[IM[i]][j - 1]
    IMx = [dict_im]

    # Load and apply scaling
    with open('DataSet/general_scaler.pkl', 'rb') as f:
        general_scaler = pickle.load(f)

    df_IMx = pd.DataFrame(IMx)
    X_new_scaled = aplicar_escalamiento(df_IMx, general_scaler)

    data_IM = {
        'nombre': signal_data['nombre'],
        'estructura': signal_data['estructura'],
        'falla': signal_data['falla'],
        'piso': signal_data['piso'],
        'IM': df_IMx
    }

    return data_IM, X_new_scaled


# %%
def load_models(tipo='det'):
    """Load all trained models and their respective scalers"""
    models = {}

    # Load Random Forest
    try:
        # loaded_scaler = joblib.load('scaler.pkl')
        # loaded_model = joblib.load('random_forest_model.pkl')
        models['Random Forest'] = {
            'model': joblib.load(f'Models/RF/RF_{tipo}.pkl'),
            'scaler': joblib.load(f'Models/RF/scaler_{tipo}.pkl')
        }
    except:
        print("Warning: Random Forest model files not found")

    # Load XGBoost
    try:
        models['XGBoost'] = {
            'model': joblib.load(f'Models/XGBoost/XGBoost_model_{tipo}.pkl'),
            'scaler': joblib.load(f'Models/XGBoost/scale_xgb_{tipo}.pkl')
        }
    except:
        print("Warning: XGBoost model files not found")

    # Load ANN
    try:
        models['Neural Network'] = {
            'model': keras.models.load_model(f'Models/ANN/modelo_nn_{tipo}.keras'),
            'scaler': joblib.load(f'Models/ANN/scaler_{tipo}.pkl')
        }

    except:
        print("Warning: Neural Network model files not found")

    return models


def predict_damage(X_new_scaled, models):
    """
    Make predictions using all available models

    Parameters:
    -----------
    X_new_scaled : pd.DataFrame
        Scaled intensity measures for new data
    models : dict
        Dictionary containing the loaded models and scalers
    """
    print("\n" + "=" * 80)
    print("STRUCTURAL DAMAGE PREDICTION REPORT")
    print("=" * 80 + "\n")

    results = []

    for model_name, model_dict in models.items():
        # Make prediction
        if model_name == 'Neural Network':
            # Neural Network prediction
            scaler_ann = model_dict['scaler']
            X_scaled = scaler_ann.transform(X_new_scaled)

            probability = model_dict['model'].predict(X_scaled)[0][0]
            prediction = [1 if probability >= 0.5 else 0]
            prob_str = f"{probability:.2%}"

        else:
            loaded_scaler = model_dict['scaler']
            loaded_model = model_dict['model']
            X_test_scaled = loaded_scaler.transform(X_new_scaled)
            prediction = loaded_model.predict(X_test_scaled)
            try:
                probability = model_dict['model'].predict_proba(X_test_scaled)[0][1]
                prob_str = f"{probability:.2%}"
            except:
                prob_str = "Not available"

        print(model_name, prediction, prob_str)
        results.append([
            model_name,
            "Damaged" if prediction[0] == 1 else "No Damage",
            prob_str
        ])

    # Print results table
    print("MODEL PREDICTIONS:")
    print("-" * 50)
    headers = ["Model", "Prediction", "Damage Probability"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("\n")

    # Consensus analysis
    predictions = [1 if row[1] == "Damaged" else 0 for row in results]
    consensus = np.mean(predictions)

    print("CONSENSUS ANALYSIS:")
    print("-" * 50)
    if consensus >= 0.66:
        consensus_result = "HIGH PROBABILITY OF DAMAGE"
    elif consensus <= 0.33:
        consensus_result = "LOW PROBABILITY OF DAMAGE"
    else:
        consensus_result = "UNCERTAIN - FURTHER INVESTIGATION RECOMMENDED"

    print(f"Model Agreement: {consensus:.0%}")
    print(f"Final Assessment: {consensus_result}")
    print("-" * 50 + "\n")

    return results, consensus


# %%
def predict_damage_location(X_new_scaled, models):
    """
    Make predictions for damage location using multiple models

    Parameters:
    -----------
    X_new_scaled : pd.DataFrame
        Scaled intensity measures for new data
    models : dict
        Dictionary containing the loaded models and scalers
    """
    print("\n" + "=" * 80)
    print("STRUCTURAL DAMAGE LOCATION PREDICTION REPORT")
    print("=" * 80 + "\n")

    results = []
    all_probabilities = []

    floor_map = {
        0: "Floor 1",
        1: "Floor 2",
        2: "Floor 3",
        3: "Floor 4"
    }

    for model_name, model_dict in models.items():
        if model_name == 'Neural Network':
            # Neural Network prediction
            scaler_ann = model_dict['scaler']
            X_scaled = scaler_ann.transform(X_new_scaled)
            probabilities = model_dict['model'].predict(X_scaled)
            prediction = np.argmax(probabilities, axis=1)[0]
            print(probabilities)
            print(prediction)
            prob_dict = {floor_map[i]: f"{prob:.2%}" for i, prob in enumerate(probabilities[0])}

        else:
            # Other models prediction
            loaded_scaler = model_dict['scaler']
            loaded_model = model_dict['model']
            X_test_scaled = loaded_scaler.transform(X_new_scaled)
            prediction = loaded_model.predict(X_test_scaled)[0]
            try:
                probabilities = loaded_model.predict_proba(X_test_scaled)[0]
                prob_dict = {floor_map[i]: f"{prob:.2%}" for i, prob in enumerate(probabilities)}
            except:
                prob_dict = {floor: "Not available" for floor in floor_map.values()}

        # Store prediction and probabilities
        results.append([
            model_name,
            floor_map[prediction],
            prob_dict[floor_map[prediction]]
        ])
        all_probabilities.append(prob_dict)

    # Print individual model predictions
    print("MODEL PREDICTIONS:")
    print("-" * 50)
    headers = ["Model", "Predicted Location", "Confidence"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("\n")

    # Print detailed probabilities for each model
    print("DETAILED PROBABILITY ANALYSIS:")
    print("-" * 50)
    prob_table = []
    # print(all_probabilities)
    for model_name, probs in zip(models.keys(), all_probabilities):
        row = [model_name]
        row.extend([probs[floor] for floor in floor_map.values()])
        prob_table.append(row)

    prob_headers = ["Model"] + list(floor_map.values())
    print(tabulate(prob_table, headers=prob_headers, tablefmt="grid"))
    print("\n")

    # Consensus analysis
    predictions = [result[1] for result in results]
    prediction_counts = {floor: predictions.count(floor) for floor in floor_map.values()}
    most_predicted = max(prediction_counts.items(), key=lambda x: x[1])
    consensus_percentage = most_predicted[1] / len(predictions)

    print("CONSENSUS ANALYSIS:")
    print("-" * 50)
    if consensus_percentage >= 0.66:
        consensus_result = f"HIGH CONFIDENCE: {most_predicted[0]}"
    elif consensus_percentage >= 0.5:
        consensus_result = f"MODERATE CONFIDENCE: {most_predicted[0]}"
    else:
        consensus_result = "UNCERTAIN - MIXED PREDICTIONS"

    print(f"Model Agreement: {consensus_percentage:.0%}")
    print(f"Final Assessment: {consensus_result}")

    # Print prediction distribution
    print("\nPrediction Distribution:")
    for floor, count in prediction_counts.items():
        print(f"{floor}: {count}/{len(predictions)} models")
    print("-" * 50 + "\n")

    return results, prediction_counts, all_probabilities


def ubicar_falla(data_IM):
    models_loc = load_models(tipo='loc')
    # Load and apply scaling
    with open('DataSet/loc_scaler.pkl', 'rb') as f:
        loc_scaler = pickle.load(f)

    X_new_scaled_2 = aplicar_escalamiento(data_IM['IM'], loc_scaler)

    results, prediction_counts, probabilities = predict_damage_location(X_new_scaled_2, models_loc)

    return results, prediction_counts, probabilities