# Cancerous-Cells-Segmentation-U-NET
This repository contains a U-Net model trained for image segmentation using the Kaggle lizard dataset and additional data from CPM-15 and CPM-17 datasets. The U-Net architecture in this implementation consists of 26 convolutional layers, providing a deep and complex network capable of capturing intricate image features. The model has achieved a validation accuracy of 91 percent, indicating its effectiveness in segmenting lizard images.

To train the model, a T4 GPU was utilized, taking advantage of its powerful parallel processing capabilities to accelerate the training process. The T4 GPU provided significant computational power, enabling faster iterations and reduced training times.

Please note that while the model's performance is promising, the Kaggle lizard dataset is a publicly available dataset, and CPM-15 and CPM-17 datasets are not widely recognized standard datasets. Therefore, it is essential to consider the training data's quality, diversity, and representativeness when interpreting the model's performance.

For more details on the architecture, training procedure, and how to use the trained model, please refer to the documentation and code provided in this repository.

# Pre-requisites
-tensorflow
-keras
-Opencv
-matplotlib
-flask

# Commands to Run

pip install flask

create venv

flask run



# Architecture of U-NET Model

![Screenshot 2023-07-11 144037](https://github.com/Hassan-293/Cancerous-Cells-Segmentation/assets/88833393/60bb6529-4114-4e80-85f5-f16bee2ec592)
