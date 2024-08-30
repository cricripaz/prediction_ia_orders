from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import calendar
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Cargar el modelo y el escalador
model = joblib.load('final/test_final/modelo_87.pkl')
scaler = joblib.load('final/test_final/scaler_87.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        year = int(request.args.get('year'))
        month = int(request.args.get('month'))

        # Generar fechas para el mes y año dados
        num_days = calendar.monthrange(year, month)[1]
        dates = [datetime(year, month, day) for day in range(1, num_days + 1)]

        # Crear DataFrame con las características para las predicciones
        features = pd.DataFrame({
            'ds': [date.toordinal() for date in dates],
            'day_of_week': [date.weekday() for date in dates],
            'month': [date.month for date in dates],
            'quarter': [(date.month - 1) // 3 + 1 for date in dates],
            'day_of_year': [date.timetuple().tm_yday for date in dates],
            'week_of_year': [date.isocalendar()[1] for date in dates]
        })

        # Escalar las características
        features_scaled = scaler.transform(features)

        # Hacer predicciones
        predictions = model.predict(features_scaled)

        # Formatear la respuesta
        result = [{
            'date': date.strftime('%Y-%m-%d'),
            'quantity': round(float(pred), 2)
        } for date, pred in zip(dates, predictions)]

        # Imprimir la respuesta en consola antes de enviarla
        print({
            'data': result,
            'message': 'Predictions get successfully',
            'year': year,
            'month': month,
        })

        return jsonify({
            'data': result,
            'message': 'Predictions get successfully',
            'year': year,
            'month': month,
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
