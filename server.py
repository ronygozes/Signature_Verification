from inference import get_loaded_model, preprocess_img
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)
model = get_loaded_model()


@app.route('/', methods=['POST'])
def index():

    """
    gets two images and decides whether they are from the same person
    :return:
    y_pred

    y_pred = 0: the same writer
    y_pred = 1: different writer
    """
    files_online = dict(request.files)
    file1 = files_online['img1']
    file2 = files_online['img2']
    img1 = Image.open(file1)
    img2 = Image.open(file2)

    img1_pro = preprocess_img(img1)
    img2_pro = preprocess_img(img2)

    im1s = np.reshape(img1_pro, (1, *img1_pro.shape, 1))
    im2s = np.reshape(img2_pro, (1, *img2_pro.shape, 1))
    y_pred = model.predict([im1s, im2s])

    if y_pred > 0.5:
        decision = 'different writer'
    else:
        decision = 'the same writer'
    return decision


@app.route('/hello')
def hello():
    return 'hello'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
