import eventlet

eventlet.monkey_patch()

from ultralytics import YOLO
from flask import Flask,render_template,request
from flask_socketio import SocketIO,emit,send
from flask_cors import CORS
import cv2
import base64
import numpy as np
import prePos as pp
from pathlib import Path
import time
#from gevent import monkey

#monkey.patch_all()
model_paths = [
    "./detect/train14/weights/best.pt",
    "./detect/train18/weights/best.pt",
    "./detect/train19/weights/best.pt",
    "./detect/train20/weights/best.pt",
    "./detect/train21/weights/best.pt",
]

app = Flask(__name__)
CORS(app,resources={r"/*": {"origins": "*"}})
model_path = model_paths[0]
model = YOLO(Path(model_path).absolute())
socket = SocketIO(app,cors_allowed_origins="*",compression=True,async_mode='eventlet', max_http_buffer_size=10_000_000)  # Set max size to 10 MB
# Increase WebSocket message limit in eventlet
eventlet.wsgi.MAX_HEADER_LINE = 8192 * 10  # Adjust based on needs

@socket.on("connect")
def connect():
    print('connected')

@socket.on("liveFeed")
def liveFeed(base64_string,prePos):
    global model_path, model_paths, model
    print('chegou')
    header, base64_data = base64_string.split(',', 1)
    if(base64_data is not None):
        image_bytes = base64.b64decode(base64_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        print('pre')
        if image is not None:
            if prePos == 'Canny-Bilateral-1024':
                image = pp.bilateral(image)
                image = pp.resize(image,1024,1024)
                image = pp.canny(image)
                if(model_path != model_paths[4]):
                    model_path = model_paths[4]
                    model = YOLO(Path(model_path).absolute())
                    time.sleep(3)


            elif prePos == 'Canny-Bilateral-640':
                image = pp.bilateral(image)
                image = pp.resize(image,640,640)
                image = pp.canny(image)
                if(model_path != model_paths[3]):
                    model_path = model_paths[3]
                    model = YOLO(Path(model_path).absolute())
                    time.sleep(3)


            elif prePos == 'Sobel-Bilateral-1024':
                image = pp.bilateral(image)
                image = pp.resize(image,1024,1024)
                image = pp.sobel(image)
                if(model_path != model_paths[2]):
                    model_path = model_paths[2]
                    model = YOLO(Path(model_path).absolute())
                    time.sleep(3)


            elif prePos == 'Sobel-Bilateral-640':
                image = pp.bilateral(image)
                image = pp.resize(image,640,640)
                image = pp.sobel(image)
                if(model_path != model_paths[1]):
                    model_path = model_paths[1]
                    model = YOLO(Path(model_path).absolute())
                    time.sleep(3)
            
            elif prePos == 'normal':
                if(model_path != model_paths[0]):
                    model_path = model_paths[0]
                    model = YOLO(Path(model_path).absolute())
                    time.sleep(3)
            print(model_path)
            cv2.imwrite("teste.png",image)
            results = model.predict(source=image, save=True)

            _, buffer = cv2.imencode('.png', results[0].plot())
            return_string = base64.b64encode(buffer).decode('utf-8')
            print('retorna')
            emit('imagemRetorno', return_string, broadcast=False)
"""             
            elif prePos == 'Laplace-Bilateral-1024':
                image = pp.bilateral(image)
                image = pp.resize(image,1024,1024)
                image = pp.laplaciano(image)

            elif prePos == 'Laplace-Bilateral-640':
                image = pp.bilateral(image)
                image = pp.resize(image,640,640)
                image = pp.laplaciano(image) """


@socket.on_error()
def debug_disconnect(e):
    print(f"ERRO: {e}" )
if __name__ == '__main__':
    socket.run(app,debug=True)