from flask import Flask,render_template,request
from flask_socketio import SocketIO,emit
from flask_cors import CORS
import cv2
import base64
import numpy as np
import os
import time

app = Flask(__name__)
CORS(app)
socket = SocketIO(app,cors_allowed_origins=["https://localhost:3000"])

@app.route('/')
def index():
    return render_template('index.html')

@socket.on("liveFeed")
def liveFeed(base64_string):
    #print('chegou')
    # Remove a parte 'data:image/png;base64,' se houver
    header, base64_data = base64_string.split(',', 1)
    if(base64_data is not None):
        # Decodifica a imagem base64 para bytes
        image_bytes = base64.b64decode(base64_data)

        # Converte os bytes em um array NumPy
        image_array = np.frombuffer(image_bytes, np.uint8)

        # Decodifica o array para uma imagem
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is not None:
            cv2.imwrite(f'./img-{int(time.time() * 1000)}.png ',image)
            #print('salvou')
            print('emitiu')
            emit('imagemRetorno',base64_data,broadcast=True)

if __name__ == '__main__':
    socket.run(app, debug=True)