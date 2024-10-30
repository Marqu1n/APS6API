from flask import Flask,render_template,request
from flask_socketio import SocketIO,emit
from flask_cors import CORS
import cv2
import base64
import numpy as np
import prePos as pp
#import eventlet

#eventlet.monkey_patch()

app = Flask(__name__)
CORS(app)
socket = SocketIO(app,cors_allowed_origins=["https://localhost:3000",'*','https://192.168.100.12:3000'])


@socket.on("liveFeed")
def liveFeed(base64_string,prePos):
    print('chegou')
    header, base64_data = base64_string.split(',', 1)
    if(base64_data is not None):
        image_bytes = base64.b64decode(base64_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        print('pre')
        if image is not None:
            if prePos == 'canny':
                image = pp.canny(image)
            elif prePos == 'sobel':
                image = pp.sobel(image)
            elif prePos == 'bilateral':
                image = pp.bilateral(image)
            elif prePos == 'cinza':
                image = pp.gray(image)
            elif prePos == 'clahe':
                image = pp.clahe(image)
            _, buffer = cv2.imencode('.png', image)
            return_string = base64.b64encode(buffer).decode('utf-8')
            print('retorna')
            emit('imagemRetorno', return_string, broadcast=True)

if __name__ == '__main__':
    socket.run(app,debug=True)