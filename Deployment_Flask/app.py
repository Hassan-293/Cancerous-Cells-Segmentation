from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from albumentations import (Compose, OneOf, 
                            CLAHE, Flip,Rotate,Transpose,ShiftScaleRotate,IAAPiecewiseAffine,RandomRotate90,ChannelShuffle,ElasticTransform,Flip,GridDistortion,HorizontalFlip,HueSaturationValue,OpticalDistortion,
                            RandomBrightnessContrast,RandomGamma,RandomSizedCrop,VerticalFlip,RGBShift,GaussNoise )
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image from the form
        image = request.files['image']
        img = Image.open(image)

        # Convert the image to a NumPy array
        

        # Perform your desired function on the image array
        # ...
        processed_img_array=func(img)

        # Convert the processed image array back to PIL image
        fig, ax = plt.subplots(1, figsize=(10, 5))
        
        ax.imshow(processed_img_array, cmap='coolwarm')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig('static/image_and_mask.png', dpi=100, bbox_inches='tight')
       


        
        # Return the processed image file path as JSON response
        return render_template('index.html',context='static/image_and_mask.png')

    return render_template('index.html')

def read_image(file_loc):
    img = Image.open(file_loc)
    img = np.array(img)
    return img

def func(image):
    
    
    # Resize the image to 256x256 pixels
    resized_image = image.resize((256, 256))
    temp_file = 'static/1.png'
    resized_image.save(temp_file)
    test_generator = Test_Generator(['static/1.png'], 1)
    from keras.models import load_model
    custom_objects = {'bce_dice_loss': 0.74,
                    'iou':0.9}


    model = load_model('static/my_model.h5',custom_objects=custom_objects)
    x_test = []
    p_test = []

    for x in test_generator:
        p = model.predict(x)
        x_test.append(np.squeeze(x, 0))
        p_test.append(np.squeeze(p, 0))
        x_test = np.array(x_test) # Image
    p_test = np.array(p_test) # Predicted mask

    

    return np.squeeze(p_test[0], -1)
class Test_Generator(Sequence):

  def __init__(self, x_set, batch_size=10, img_dim=(256,256), augmentation=False):
      self.x = x_set
      self.batch_size = batch_size
      self.img_dim = img_dim
      self.augmentation = augmentation

  def __len__(self):
      return math.ceil(len(self.x) / self.batch_size)

  aug = Compose(
    [
      CLAHE(always_apply=True, p=1.0)
    ])


  def __getitem__(self, idx):
      batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

      # batch_x = np.array([cv2.resize(cv2.cvtColor(cv2.imread(file_name, -1), cv2.COLOR_BGR2RGB), (512, 512)) for file_name in batch_x])
      # batch_y = np.array([cv2.resize(cv2.cvtColor(cv2.imread(file_name, -1), cv2.COLOR_BGR2RGB), (512, 512)) for file_name in batch_y])
      batch_x = np.array([read_image(file_name) for file_name in batch_x])

      if self.augmentation is True:
        aug = [self.aug(image=i) for i in batch_x]
        batch_x = np.array([i['image'] for i in aug])

      return batch_x/255.0

if __name__ == '__main__':
    app.run(debug=True)

