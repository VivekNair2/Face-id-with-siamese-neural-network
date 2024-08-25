from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
from model import siamese_model, verify

app = Flask(__name__)
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[80:80+250, 200:200+250, :]
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/verify', methods=['POST'])
def verify_image():
    ret, frame = cap.read()
    frame = frame[80:80+250, 200:200+250, :]
    cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
    results, verified = verify(siamese_model, 0.9, 0.7)
    return {'verified': verified}

if __name__ == '__main__':
    app.run(debug=True)
