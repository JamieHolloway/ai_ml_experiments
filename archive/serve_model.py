from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = joblib.load(r'E:\OneDrive\repo\sandbox\AI_ML\models\iris_model.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = data.get('input')

        if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
            return jsonify({'error': 'Input must be a list of numbers'}), 400

        prediction = model.predict(np.array(features).reshape(1, -1))
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
