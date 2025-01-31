from flask import Flask, render_template, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configuration pour le chargement des images
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app, photos)

# Charger le modèle de machine learning
model = pickle.load(open('model.pickle', 'rb'))
cols = ['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']

@app.route('/')
def index():
    return render_template('index.html')

#route pour le modèle de machine learning
@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final = np.array(features)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = model.predict(data_unseen)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='CO2 Emission of the vehicle is {}'.format(output))


#route pour le modèle de deep learning (resnet50)
@app.route('/ex2', methods=['GET', 'POST'])
def ex2():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        filepath = f'./static/img/{filename}'
        
        
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        
        resnet_model = ResNet50(weights='imagenet')
        preds = resnet_model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=1)[0][0]
        
        return render_template('ex2.html', image_name=filename, prediction_text=f'Classe prédite : {decoded_preds[1]}, Confiance : {decoded_preds[2]:.2f}')
    
    return render_template('ex2.html')


# cv et clf pour les emails
@app.route('/ex3', methods=['GET', 'POST'])
def ex3():
    if request.method == 'POST':
        cv = pickle.load(open('./models/cv.pkl', 'rb'))
        clf = pickle.load(open('./models/clf.pkl', 'rb'))
        
        email = request.form['email']
        tokenized_email = cv.transform([email])
        prediction = clf.predict(tokenized_email)
        
        return render_template('ex3.html', prediction_text='Spam' if prediction[0] == 1 else 'Non Spam')
    
    return render_template('ex3.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST':
        cv=pickle.load(open('./models/cv.pkl','rb'))
        clf=pickle.load(open('./models/clf.pkl','rb'))

        email = request.json['email']
        tokenized_email = cv.transform([email])
        prediction = clf.predict(tokenized_email)
        
        return jsonify({'prediction': 'Spam' if prediction[0] == 1 else 'Non Spam'})
    

if __name__ == '__main__':
    app.run()