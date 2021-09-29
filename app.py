from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np

import cv2
import numpy as np
from flask import Flask,request,jsonify,render_template
from PIL import Image
def load_image(img_pa):
    
   
    class_name=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    img = Image.open(img_pa)
    img = np.array(img)
    
    img=cv2.resize(img,(32,32))

    img=np.expand_dims(img, axis=0)
    
    img=img/255.
    
    cifer_model=load_model('cifer_Model.hdf5')
    
    pred=cifer_model.predict(img)
    
    index_Name=np.argmax(pred[0])
    
    return class_name[index_Name]




app=Flask(__name__)

@app.route('/')
def correct():
    return render_template('Cifer.html')


@app.route('/successCifer', methods = ['POST'])  

def success():  
    if request.method == 'POST':  
        f = request.files['file']  
       
        r=load_image(f)
        
        return render_template("successCifer.html", name = r)  
    
if __name__ == '__main__':  
    app.run()  
