from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json

app = Flask(__name__)


def get_result(arr):

    names = [
        '(vertigo) Paroymsal Positional Vertigo', 'AIDS', 'Acne',
        'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
        'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
        'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)',
        'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis',
        'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
        'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
        'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
        'Osteoarthristis', 'Paralysis (brain hemorrhage)',
        'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
        'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A'
    ]
    # if np.any(arr):
    #     arr = arr.reshape(1, 132)


    arr = np.array(arr).reshape(1, 132)
    new_model = tf.keras.models.load_model('disease_prediction.h5')

    prediction = new_model.predict(arr)
    index = np.argmax(prediction)

    disease_name = names[index]
    probability =float( np.max(new_model.predict(arr)))

    return {"disease_name": disease_name, "probability": probability}


@app.route('/predict', methods=['POST'])
def predict_disease():
    data = request.get_json()

    if not data or not 'symptoms' in data:
        return jsonify({'message': 'Missing required field: "symptoms"'}), 400

    symptoms = data['symptoms']

    result = get_result(symptoms)

    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True)
