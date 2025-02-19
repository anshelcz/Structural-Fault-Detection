"""
Title: File of functions about seismic signal processing
Author: Anshel Chuquiviguel Zaña
Contact: anshel.chuquiviguel@utec.edu.pe
Created: February 19, 2025
Last Updated: February 19, 2025
Version: 1.0
"""
import numpy as np
from scipy import integrate, signal, stats
import pywt
import warnings
warnings.filterwarnings("ignore")

def aplicar_promedio_movil(senal, ventana=50):
    """
    Aplica un filtro de promedio móvil ponderado a la señal.

    Parámetros:
    senal : array_like
        Serie temporal de aceleraciones
    ventana : int
        Tamaño de la ventana del promedio móvil

    Retorna:
    array_like : Señal suavizada
    """
    # Crear kernel de ponderación (triangular)
    kernel = np.concatenate([np.arange(1, ventana // 2 + 1),
                             np.arange((ventana - 1) // 2, 0, -1)])
    kernel = kernel / np.sum(kernel)

    # Aplicar convolución
    return np.convolve(senal, kernel, mode='same')


def analisis_wavelet(senal, dt, wavelet='db4', nivel=10):
    """
    Realiza el análisis wavelet de la señal y calcula parámetros energéticos.

    Parámetros:
    senal : array_like
        Serie temporal de aceleraciones
    dt : float
        Intervalo de tiempo entre muestras
    wavelet : str
        Tipo de wavelet a utilizar (default: 'db4')
    nivel : int
        Nivel de descomposición wavelet

    Retorna:
    dict : Parámetros energéticos basados en wavelets
    """
    # Realizar la descomposición wavelet
    coeficientes = pywt.wavedec(senal, wavelet, level=nivel)

    # Calcular energía por nivel
    energias = {}
    energia_total = 0

    # Energía de los coeficientes de aproximación (último nivel)
    energia_aprox = np.sum(coeficientes[0] ** 2)
    energias['aprox'] = energia_aprox
    energia_total += energia_aprox

    # Energía de los coeficientes de detalle
    for i, coef in enumerate(coeficientes[1:], 1):
        energia = np.sum(coef ** 2)
        energias[f'detalle_{i}'] = energia
        energia_total += energia

    # Calcular porcentajes de energía
    resultados = {
        'energia_total': energia_total,
        'energia_aprox_norm': energia_aprox / energia_total
    }

    # for i in range(1, nivel + 1):
    #    resultados[f'energia_detalle_{i}_norm'] = energias[f'detalle_{i}']/energia_total

    # Calcular bandas de frecuencia aproximadas
    # fs = 1/dt
    # freq_bands = []
    # for i in range(nivel):
    #    freq_sup = fs/(2**(i+1))
    #    freq_inf = fs/(2**(i+2))
    #    freq_bands.append((freq_inf, freq_sup))

    # resultados['freq_bands'] = freq_bands

    return resultados


def calcular_medidas_intensidad(senal, dt, aplicar_filtro=True):
    """
    Calcula múltiples medidas de intensidad para una señal sísmica.

    Parámetros:
    senal : array_like
        Serie temporal de aceleraciones
    dt : float
        Intervalo de tiempo entre muestras

    Retorna:
    dict : Diccionario con las diferentes medidas de intensidad
    """
    # Convertir a array de numpy si no lo es
    senal = np.array(senal)
    if aplicar_filtro:
        senal = aplicar_promedio_movil(senal)

    # 1. PGA (Peak Ground Acceleration)
    pga = np.max(np.abs(senal))

    # 2. PGV (Peak Ground Velocity)
    velocidad = integrate.cumtrapz(senal, dx=dt, initial=0)
    pgv = np.max(np.abs(velocidad))

    # 3. PGD (Peak Ground Displacement)
    desplazamiento = integrate.cumtrapz(velocidad, dx=dt, initial=0)
    pgd = np.max(np.abs(desplazamiento))

    # 4. Intensidad de Arias (IA)
    ia = np.pi / (2 * 9.81) * integrate.trapz(senal ** 2, dx=dt)

    # 5. CAV (Cumulative Absolute Velocity)
    cav = integrate.trapz(np.abs(senal), dx=dt)

    # 6. RMS de la aceleración
    arms = np.sqrt(np.mean(senal ** 2))

    # 7. Duración significativa (tiempo entre 5% y 95% de IA)
    ia_tiempo = np.cumsum(senal ** 2) * dt * np.pi / (2 * 9.81)
    ia_normalizado = ia_tiempo / ia_tiempo[-1]
    t_5 = np.where(ia_normalizado >= 0.05)[0][0] * dt
    t_95 = np.where(ia_normalizado >= 0.95)[0][0] * dt
    duracion_significativa = t_95 - t_5

    # 8. Espectro de Fourier
    freq, power = signal.welch(senal, fs=1 / dt)
    freq_predominante = freq[np.argmax(power)]

    # 9. Intensidad espectral de Housner
    si = integrate.trapz(np.abs(velocidad), dx=dt)

    # 10. Parámetros energéticos mediante wavelets
    params_wavelet = analisis_wavelet(senal, dt)

    # 11. Energía total de la señal en dominio del tiempo
    energia_tiempo = np.sum(senal ** 2) * dt

    # Combinar todos los resultados
    resultados = {
        'PGA': pga,
        'PGV': pgv,
        'PGD': pgd,
        'Intensidad_Arias': ia,
        'CAV': cav,
        'RMS_Aceleracion': arms,
        'Duracion_Significativa': duracion_significativa,
        'Frecuencia_Predominante': freq_predominante,
        'Intensidad_Housner': si,
        'Energia_Tiempo': energia_tiempo
    }

    # Agregar parámetros wavelet
    resultados.update({
        f'Energia_Wavelet_{k}': v
        for k, v in params_wavelet.items()
        if k != 'freq_bands'
    })

    return resultados


def analizar_piso(senales_piso, dt=1 / 1000, aplicar_filtro=True):
    """
    Analiza todas las señales de un piso y devuelve estadísticas agregadas.

    Parámetros:
    senales_piso : list
        Lista de arrays con las señales del piso
    dt : float
        Intervalo de tiempo entre muestras

    Retorna:
    dict : Estadísticas agregadas de las medidas de intensidad
    """
    resultados = []
    for i in range(1, 6):
        resultados.append(calcular_medidas_intensidad(senales_piso[:, i], dt, aplicar_filtro))

    # Calcular estadísticas agregadas
    estadisticas = {}
    for medida in resultados[0].keys():
        valores = [r[medida] for r in resultados]
        estadisticas[medida] = np.array(valores)

    return estadisticas
