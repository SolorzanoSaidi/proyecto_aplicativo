from typing import Any
from rest_framework.views import APIView
from rest_framework import serializers
from rest_framework.response import Response
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn
from skfuzzy import control as ctrl

import skfuzzy as fuzz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
print(sklearn.__version__)


model = tf.keras.models.load_model('core/model/red_neuronal.h5')
scaler = joblib.load('core/model/scaler.joblib')
rf_model = joblib.load('core/model/rf_model.joblib')


class Predict(APIView):

    class InputSerializer(serializers.Serializer):
        temperature = serializers.FloatField()
        humidity = serializers.FloatField()
        tds = serializers.FloatField()
        ph = serializers.FloatField()

    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.simulacion = None

        self.init_data()
        self.modile_hibrido_difuso_neutrosofico()

    def calidad_optima(self, row):
        if (18 <= row['Temperature (°C)'] <= 25) and (6.0 <= row['pH Level'] <= 6.5) and (700 <= row['TDS Value (ppm)'] <= 800) and (60 <= row['Humidity (%)'] <= 70):
            return 1
        else:
            return 0

    def init_data(self):
        self.data = pd.read_csv(
            'core/data/lettucedataset.csv', delimiter=',', encoding='latin1')
        self.data.isnull().sum()
        self.data = self.data.drop(columns=['Plant_ID', 'Date'])

        self.data['calidad'] = self.data.apply(self.calidad_optima, axis=1)
        X = self.data[['Temperature (°C)', 'Humidity (%)',
                       'TDS Value (ppm)', 'pH Level']].values

        y = self.data['calidad'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def valor_neutorosofo(self, calidad_final_decimal):
        if calidad_final_decimal >= 0 and calidad_final_decimal < 0.10:
            valor_neutrosófico = "em"

        elif calidad_final_decimal >= 0.10 and calidad_final_decimal < 0.20:
            valor_neutrosófico = "mmm"
        elif calidad_final_decimal >= 0.20 and calidad_final_decimal < 0.30:
            valor_neutrosófico = "mm"
        elif calidad_final_decimal >= 0.30 and calidad_final_decimal < 0.40:
            valor_neutrosófico = "ma"
        elif calidad_final_decimal >= 0.40 and calidad_final_decimal < 0.50:
            valor_neutrosófico = "mdm"
        elif calidad_final_decimal >= 0.50 and calidad_final_decimal < 0.60:
            valor_neutrosófico = "m"
        elif calidad_final_decimal >= 0.60 and calidad_final_decimal < 0.70:
            valor_neutrosófico = "mdb"
        elif calidad_final_decimal >= 0.70 and calidad_final_decimal < 0.80:
            valor_neutrosófico = "b"
        elif calidad_final_decimal >= 0.80 and calidad_final_decimal < 0.90:
            valor_neutrosófico = "mb"
        elif calidad_final_decimal >= 0.90 and calidad_final_decimal < 1:
            valor_neutrosófico = "mmb"
        elif calidad_final_decimal >= 1:
            valor_neutrosófico = "eb"

        return valor_neutrosófico

    def modile_hibrido_difuso_neutrosofico(self):

        # Definición de las variables de entrada y salida
        temperature = ctrl.Antecedent(
            np.arange(0, 33.6, 0.1), 'Temperature (°C)')

        humidity = ctrl.Antecedent(np.arange(0, 81, 0.1), 'Humidity (%)')
        tds = ctrl.Antecedent(np.arange(0, 801, 0.1), 'TDS Value (ppm)')
        ph = ctrl.Antecedent(np.arange(0, 7, 0.1), 'pH Level')

        # Definición de la variable de salida
        calidad = ctrl.Consequent(np.arange(0, 11, 0.1), 'calidad')

        # Funciones de membresía para Temperature
        temperature['baja'] = fuzz.trapmf(temperature.universe, [0, 0, 18, 22])
        temperature['media'] = fuzz.trimf(temperature.universe, [20, 24, 28])
        temperature['alta'] = fuzz.trapmf(
            temperature.universe, [26, 30, 33.5, 33.5])

        # Funciones de membresía para Humidity
        humidity['baja'] = fuzz.trapmf(humidity.universe, [0, 0, 50, 60])
        humidity['media'] = fuzz.trimf(humidity.universe, [55, 65, 75])
        humidity['alta'] = fuzz.trapmf(humidity.universe, [65, 75, 80, 80])

        # Funciones de membresía para TDS
        tds['baja'] = fuzz.trapmf(tds.universe, [0, 0, 400, 500])
        tds['media'] = fuzz.trimf(tds.universe, [450, 600, 800])
        tds['alta'] = fuzz.trapmf(tds.universe, [600, 700, 800, 800])

        # Funciones de membresía para pH
        ph['baja'] = fuzz.trapmf(ph.universe, [0, 0, 6, 6.3])
        ph['media'] = fuzz.trimf(ph.universe, [6.2, 6.5, 6.7])
        ph['alta'] = fuzz.trapmf(ph.universe, [6.5, 6.7, 6.9, 6.9])

        # Definición de las funciones de membresía para calidad
        calidad['baja'] = fuzz.trapmf(calidad.universe, [0, 0, 2, 4])
        calidad['media'] = fuzz.trimf(calidad.universe, [3, 5, 7])
        calidad['alta'] = fuzz.trapmf(calidad.universe, [6, 8, 10, 10])

        # Definición de las reglas difusas
        reglas = [
            ctrl.Rule(temperature['alta'] & humidity['baja'] &
                      tds['baja'] & ph['baja'], calidad['media']),
            ctrl.Rule(temperature['alta'] & humidity['baja'] &
                      tds['alta'] & ph['alta'], calidad['media']),
            ctrl.Rule(temperature['alta'] & humidity['media'] &
                      tds['media'] & ph['media'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['media'] &
                      tds['media'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['alta'] &
                      tds['alta'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['alta'] &
                      tds['alta'] & ph['baja'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['alta'] &
                      tds['alta'] & ph['media'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['media'] &
                      tds['baja'] & ph['baja'], calidad['baja']),
            ctrl.Rule(temperature['alta'] & humidity['media'] &
                      tds['baja'] & ph['media'], calidad['media']),
            ctrl.Rule(temperature['alta'] & humidity['media'] &
                      tds['baja'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['media'] &
                      tds['alta'] & ph['media'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['media'] &
                      tds['alta'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['baja'] &
                      tds['media'] & ph['baja'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['baja'] &
                      tds['media'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['alta'] & humidity['baja'] &
                      tds['media'] & ph['media'], calidad['media']),
            ctrl.Rule(temperature['alta'] & humidity['alta'] &
                      tds['media'] & ph['alta'], calidad['media']),
            ctrl.Rule(temperature['alta'] & humidity['alta'] &
                      tds['media'] & ph['baja'], calidad['media']),
            ctrl.Rule(temperature['alta'] & humidity['alta'] &
                      tds['media'] & ph['media'], calidad['media']),
            ctrl.Rule(temperature['alta'] & humidity['alta'] &
                      tds['media'] & ph['media'], calidad['media']),
            ctrl.Rule(temperature['alta'] & humidity['alta'] &
                      tds['baja'] & ph['baja'], calidad['baja']),
            ctrl.Rule(temperature['alta'] & humidity['media'] &
                      tds['media'] & ph['baja'], calidad['baja']),
            ctrl.Rule(temperature['media'] & humidity['baja'] &
                      tds['media'] & ph['baja'], calidad['media']),
            ctrl.Rule(temperature['media'] & humidity['media'] &
                      tds['baja'] & ph['media'], calidad['media']),
            ctrl.Rule(temperature['media'] & humidity['alta'] &
                      tds['alta'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['media'] & humidity['baja'] &
                      tds['alta'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['media'] & humidity['media'] &
                      tds['media'] & ph['media'], calidad['media']),
            ctrl.Rule(temperature['media'] & humidity['baja'] &
                      tds['baja'] & ph['alta'], calidad['media']),
            ctrl.Rule(temperature['media'] & humidity['alta'] &
                      tds['media'] & ph['baja'], calidad['media']),
            ctrl.Rule(temperature['media'] & humidity['media'] &
                      tds['baja'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['media'] & humidity['alta'] &
                      tds['media'] & ph['media'], calidad['alta']),
            ctrl.Rule(temperature['media'] & humidity['baja'] &
                      tds['alta'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['baja'] & humidity['baja'] &
                      tds['baja'] & ph['baja'], calidad['baja']),
            ctrl.Rule(temperature['baja'] & humidity['media'] &
                      tds['media'] & ph['media'], calidad['media']),
            ctrl.Rule(temperature['baja'] & humidity['alta'] &
                      tds['alta'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['baja'] & humidity['media'] &
                      tds['alta'] & ph['baja'], calidad['media']),
            ctrl.Rule(temperature['baja'] & humidity['baja'] &
                      tds['media'] & ph['media'], calidad['media']),
            ctrl.Rule(temperature['baja'] & humidity['alta'] &
                      tds['alta'] & ph['media'], calidad['alta']),
            ctrl.Rule(temperature['baja'] & humidity['baja'] &
                      tds['media'] & ph['alta'], calidad['media']),
            ctrl.Rule(temperature['baja'] & humidity['baja'] &
                      tds['alta'] & ph['baja'], calidad['media']),
            ctrl.Rule(temperature['baja'] & humidity['media'] &
                      tds['baja'] & ph['media'], calidad['media']),
            ctrl.Rule(temperature['baja'] & humidity['alta'] &
                      tds['baja'] & ph['alta'], calidad['alta']),
            ctrl.Rule(temperature['baja'] & humidity['alta'] &
                      tds['baja'] & ph['media'], calidad['alta']),
        ]

        # Creación del sistema de control difuso
        sistema_control = ctrl.ControlSystem(reglas)

        # Simulación del sistema de control
        self.simulacion = ctrl.ControlSystemSimulation(sistema_control)

    def post(self, request):

        data = self.InputSerializer(data=request.data)
        if data.is_valid():
                temperature = data.validated_data.get('temperature')
                humidity = data.validated_data.get('humidity')
                tds = data.validated_data.get('tds')
                ph = data.validated_data.get('ph')

                # Cargar los datos y dividirlos en conjuntos de entrenamiento y prueba
                nuevas_muestras = [[temperature, humidity, tds, ph]]

                nuevas_muestras = np.array(nuevas_muestras)

                nueva_muestra_scaled = scaler.transform(nuevas_muestras)

                nuevas_predicciones_prob_result = model.predict(
                nueva_muestra_scaled)
                # Probabilidad de la clase positiva
                nueva_predic_prob = nuevas_predicciones_prob_result[0][0]

                # Añadir la nueva muestra a X_test y su etiqueta hipotética
                X_combined = np.vstack([self.X_test, nuevas_muestras])
                y_combined = np.concatenate(
                [self.y_test, [0 if nueva_predic_prob < 0.5 else 1]])

                # Calcular las probabilidades para el conjunto combinado
                y_predic_prob_combined = model.predict(
                scaler.transform(X_combined)).ravel()

                # Calcular la curva ROC
                fpr, tpr, thresholds = roc_curve(
                y_combined, y_predic_prob_combined)

                roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
                }

                # Devolver la predicción para la nueva muestra
                response_data_rn = {
                'roc_data': roc_data,
                'nueva_prediccion': (nueva_predic_prob >= 0.5).astype(int).tolist(),
                'nueva_prediccion_probabilidad': nuevas_predicciones_prob_result.tolist()
                }

                y_pred = rf_model.predict(nueva_muestra_scaled)

                y_pred_prob = rf_model.predict_proba(nueva_muestra_scaled)[:, 1]

                # combinar para hacer curva de roc
                X_combined_rf = np.vstack([self.X_test, nuevas_muestras])
                y_combined_rf = np.concatenate(
                [self.y_test, y_pred])

                y_pred_prob_combined_rf = rf_model.predict_proba(
                X_combined_rf)[:, 1]

                fpr_rf, tpr_rf, thresholds_rf = roc_curve(
                y_combined_rf, y_pred_prob_combined_rf)

                response_data_rf = {
                'roc_data': {
                        'fpr': fpr_rf.tolist(),
                        'tpr': tpr_rf.tolist(),
                        'thresholds': thresholds_rf.tolist()
                },
                'nueva_prediccion': y_pred.tolist(),
                'nueva_prediccion_probabilidad': y_pred_prob.tolist()
                }
                print(temperature, humidity, tds, ph)
                self.simulacion.input['Temperature (°C)'] = float(temperature)
                self.simulacion.input['Humidity (%)'] = float(humidity)
                self.simulacion.input['TDS Value (ppm)'] = float(tds)
                self.simulacion.input['pH Level'] = float(ph)
                print(self.simulacion.input)
                self.simulacion.compute()

                calidad_final = self.simulacion.output['calidad']
                
                calidad_final_decimal = calidad_final * 0.1

                accuracy_rf = accuracy_score(
                self.y_test, rf_model.predict(self.X_test))
                precision_rf = precision_score(
                self.y_test, rf_model.predict(self.X_test))
                recall_rf = recall_score(self.y_test, rf_model.predict(self.X_test))
                f1_rf = f1_score(self.y_test, rf_model.predict(self.X_test))

                


                response = {
                'red_neuronal': response_data_rn,
                'random_forest': response_data_rf,
                'calidad_final': calidad_final_decimal,
                'valor_neutrosófico': self.valor_neutorosofo(calidad_final_decimal),
                }
                print(response)

                return Response({'status': 'success', 'message': 'Prediction successful', 'data': response})
