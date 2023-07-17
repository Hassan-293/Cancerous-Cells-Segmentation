# Cancerous-Cells-Segmentation-U-NET
This repository contains a convolutional pre-trained U-Net model trained for image segmentation using the Kaggle lizard dataset and additional data from CPM-15 and CPM-17 datasets. The U-Net architecture in this implementation consists of 26 convolutional layers, providing a deep and complex network capable of capturing intricate image features. The model has achieved a validation accuracy of 91 percent, indicating its effectiveness in segmenting lizard images.

A T4 GPU was utilized to train the model, taking advantage of its powerful parallel processing capabilities to accelerate the training process. The T4 GPU provided significant computational power, enabling faster iterations and reduced training times.

Please note that while the model's performance is promising, the Kaggle lizard dataset is a publicly available dataset, and CPM-15 and CPM-17 datasets are not widely recognized standard datasets. Therefore, it is essential to consider the training data's quality, diversity, and representativeness when interpreting the model's performance.

For more details on the architecture, training procedure, and how to use the trained model, please refer to the documentation and code provided in this repository.

# Pre-Requisites
+ tensorflow
+ keras
+ opencv
+ matplotlib
+ flask


# Architecture of U-NET Model

The U-Net architecture consists of an encoder-decoder structure with skip connections. The encoder part is responsible for capturing and encoding the spatial information from the input image, while the decoder part aims to reconstruct the original resolution and generate the segmentation mask. The skip connections help to bridge the gap between the encoder and decoder by combining the high-resolution features from the encoder with the upsampled features from the decoder. This allows the model to retain both local and global contextual information throughout the network.

![Screenshot 2023-07-11 144037](https://github.com/Hassan-293/Cancerous-Cells-Segmentation/assets/88833393/60bb6529-4114-4e80-85f5-f16bee2ec592)

In the encoder part of the U-Net, the input image undergoes a series of convolutional and pooling layers, progressively reducing the spatial dimensions while increasing the number of channels or feature maps. This helps in learning hierarchical features at different scales. Each convolutional block is typically composed of two or more convolutional layers followed by a non-linear activation function, such as ReLU (Rectified Linear Unit), and optionally, batch normalization.

The decoder part of the U-Net involves upsampling the feature maps to the original image resolution. This is achieved through a series of up-convolutional (transpose convolution) layers, which expand the spatial dimensions while reducing the number of channels. Skip connections are then employed to concatenate the feature maps from the corresponding encoder block, allowing the decoder to incorporate high-resolution information. The concatenated feature maps are then further processed through a series of convolutional layers to refine the segmentation predictions.

At the end of the U-Net architecture, a 1x1 convolutional layer with a sigmoid activation function is typically used to generate the final segmentation mask. The sigmoid function outputs pixel-wise probabilities indicating the presence or absence of the segmented object or region in each pixel of the input image.

During training, the U-Net model is typically optimized using a loss function suited for image segmentation, such as the dice loss or binary cross-entropy loss. The model learns to minimize the chosen loss function by adjusting its parameters through backpropagation and gradient descent optimization algorithms.

One of the advantages of the U-Net architecture is its ability to handle limited training data efficiently. By utilizing skip connections, the U-Net can learn to make precise predictions even with a relatively small number of annotated images. Additionally, the U-Net architecture allows for efficient inference since the decoding pathway is symmetric to the encoding pathway.

Overall, the U-Net model has proven to be highly effective in various image segmentation tasks, particularly in medical imaging applications like tumor segmentation, cell counting, and organ segmentation. Its unique architecture, combining the encoder-decoder structure with skip connections, enables accurate and detailed segmentation results by leveraging both local and global context information.

# Datasets
+ [Lizard Dataset](https://www.kaggle.com/datasets/aadimator/lizard-dataset)
+ [CPM-15 & CPM-17](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK)


# Related Work
+ [Lizard A Large-Scale Dataset for Colonic Nuclear Instance Segmentation and Classification.pdf](https://github.com/Hassan-293/Cancerous-Cells-Segmentation/files/12067478/Lizard.A.Large-Scale.Dataset.for.Colonic.Nuclear.Instance.Segmentation.and.Classification.pdf)

+ [CoNIC Colon Nuclei Identification and Counting.pdf](https://github.com/Hassan-293/Cancerous-Cells-Segmentation/files/12067619/CoNIC.Colon.Nuclei.Identification.and.Counting.pdf)

