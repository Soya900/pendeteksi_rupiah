from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import redirect, secure_filename
import os
import numpy as np
import pickle5 as pickle
import cv2
import time
from gtts import gTTS

app = Flask(__name__)

model_knn = pickle.load(open('modelknn3_5_21.pkl', 'rb'))

UPLOAD_FOLDER = '.\static\images\prediksi'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG', 'gif', 'GIF'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def index():
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

    if request.method == 'POST':
        gambar = request.files['gambar']

        if gambar.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # if os.path.isfile(app.config['UPLOAD_FOLDER']):
        #     os.remove(f for f in os.listdir(app.config['UPLOAD_FOLDER']))

        if gambar and allowed_file(gambar.filename):
            ext = gambar.filename.split('.')
            namaGambar = secure_filename('rupiah.' + ext[-1])
            gambar.save(os.path.join(
                app.config['UPLOAD_FOLDER'], namaGambar))
            return redirect('/hasil')
    else:
        return render_template('index.html')


@app.route('/hasil')
def hasil():
    # for file in os.listdir(os.path.join(img_folder, dir1)):
    #     image_path = os.path.join(img_folder, dir1, file)
    # prediksi = predict(os.path.join(app.config['UPLOAD_FOLDER'], 'rupiah.jpg'))

    gambar = [f for f in os.listdir(app.config['UPLOAD_FOLDER'])]

    if not gambar:
        return redirect('/')
    else:
        prediksi = predict(os.path.join(
            app.config['UPLOAD_FOLDER'], gambar[0]))
        suara = umpan_balik(prediksi)

        return render_template('hasil.html', prediksi=[prediksi, os.path.join(app.config['UPLOAD_FOLDER'], gambar[0]), suara])
    # ret = os.listdir(app.config['UPLOAD_FOLDER'])
    # return jsonify(ret)


@app.route('/serviceworker.js')
def sw():
    return app.send_static_file('serviceworker.js')


def predict(uang):
    panjang, lebar = 200, 200
    # nominal = ['1000', '10000', '100000', '2000', '20000', '5000', '50000']
    nominal = ['1000', '2000', '5000', '10000', '20000', '50000', '100000']

    img = cv2.imread(uang)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (panjang, lebar), interpolation=cv2.INTER_AREA)
    img = np.array(img)
    img = img.astype('float32')
    img /= 255
    img = img.reshape(-1, 200*200*3)

    knn_pred = model_knn.predict(img)
    return nominal[knn_pred[0]]


def umpan_balik(pred, lang='id'):
    prediction = pred + 'rupiah'
    language = lang
    myobj = gTTS(text=prediction, lang=language, slow=False)
    myobj.save(os.path.join(app.config['UPLOAD_FOLDER'], 'zprediksi.mp3'))
    return os.path.join(app.config['UPLOAD_FOLDER'], 'zprediksi.mp3')


# if __name__ == "__main__":
#     app.run(host='192.168.100.11', debug=True)

if __name__ == "__main__":
    app.run(debug=True)
