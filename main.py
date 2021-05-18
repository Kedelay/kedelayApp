# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
from flask import Flask
from flask import request
from flask import render_template
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch
import cv2
import numpy as np
from keras.models import Model, load_model

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DEVICE = "cpu"
MODEL = None


def predict(image_path, model):
    I = []
    print(image_path)
    img=cv2.imread(image_path)
    img=cv2.resize(img,(100,100))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    I.append(img)
    NP_ARRAY = np.array(I)
    X_VAL = NP_ARRAY/255.0
    R = MODEL.predict_classes(X_VAL)
    #model.predict(X_VAL) > 0.5).astype("int32")
    
    LABEL=''

    if(R[0]==0):
        LABEL = 'DOWNY'
    if(R[0]==1):
        LABEL = 'HEALTY'
    if(R[0]==2):
        LABEL = 'RUST'
    
    print(LABEL)
    
    return LABEL




@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, MODEL)
            return render_template("index.html", prediction=pred, image_loc='./static/'+image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    
    MODEL=load_model('./soybean.h5')   
    
    app.run()
