# IU INTERNATIONAL UNIVERSITY OF APPLIED SCIENCES

### Project: Artificial Intelligence (DLBDSEAIS02)
Utilizing TensorFlow as the primary framework, the project focuses on emotion detection from facial expressions using a Convolutional Neural Network (CNN). TensorFlow's deployment capabilities, including TensorFlow Serving and TensorFlow Lite, ensure seamless transition to production. The workflow involves a pre-processing pipeline, transforming images to a 48x48 dimension from the mma-facial-expression dataset. The CNN architecture comprises convolution layers, max pooling, ReLU activation, dense layers, and a fully connected layer with Softmax activation. Model testing involves metrics like accuracy, recall, and precision, with the F1-score for multi-classification. Deployment is done via FastAPI and Docker, enabling local or cloud-based execution.

## Purpose
Build an AI tool for the marketing department to analyze emotional responses to company ads using facial expressions in images. Develop a system to categorize emotions (e.g., joy, anger, fear) from an image dataset. Choose relevant technologies for components and communication. The goal is to identify at least three emotional states through facial expressions.

# Development Planning - UML Schemas
## UML Diagram
Diagram (Revised from Concept Phase) - Design of Convolutional Neural Network to classify emotions and serve using API
![Design of Convolutional Neural Network to classify emotions and serve using API](/images/UML_updated.jpg)

The Keras Sequential model created is a Convolutional Neural Network (CNN) designed for image classification, more specifically the classification of 7 different emotions present in the faces. 

The architecture begins with a Convolutional Layer (Conv2D_1) with 32 filters, followed by Batch Normalization. Subsequently, Conv2D_2 employs 64 filters with ReLU activation and 'same' padding, accompanied by Batch Normalization. MaxPooling2D_1 reduces spatial dimensions, and Dropout_1 prevents overfitting. The model further includes additional pairs of Convolutional and Batch Normalization layers (Conv2D_3, BatchNormalization_3, Conv2D_4, BatchNormalization_4), each followed by MaxPooling and Dropout. The pattern continues with Conv2D_5, BatchNormalization_5, Conv2D_6, BatchNormalization_6, MaxPooling2D_3, and Dropout_3. The Flattening layer prepares the data for Dense layers, starting with Dense_1 (2048 units, ReLU activation) and BatchNormalization_7, followed by Dropout_4. The final Dense_2 layer produces the output probabilities for 7 classes, utilizing the softmax activation function. 

This architecture combines convolutional and fully connected layers, augmented by batch normalization and dropout for enhanced generalization and prevention of overfitting.

## Dataset Overview
The provided image dataset, named MMAFEDB, is organized into three main folders: test, train, and valid. Each of these folders contains subfolders for seven different emotions: angry, disgust, fear, happy, neutral, sad, and surprise. The dataset statistics reveal that the test set comprises 17,356 images, with 13,767 in grayscale and 3,589 in RGB. The training set consists of 92,968 images, with 64,259 in grayscale and 28,709 in RGB. The validation set contains 17,356 images, with 13,767 in grayscale and 3,589 in RGB. The dataset is sourced from Kaggle (https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression). Emotion-wise distribution across the sets varies, with neutral being the most prevalent emotion. 

Train Folder:
- Neutral: **29,384 images**
- Angry: 6,566 images
- Fear: *4,859 images*
- Happy: **28,592 images**
- Disgust: *3,231 images*
- Sad: **12,223 images**
- Surprise: 8,113 images

Test Folder:
- Neutral: **5,858 images**
- Angry: 1,041 images
- Fear: 691 images
- Happy: **5,459 images**
- Disgust: *655 images*
- Sad: **2,177 images**
- Surprise: 1,475 images

Valid Folder:
- Neutral: **5,839 images**
- Angry: 1,017 images
- Fear: *659 images*
- Happy: **5,475 images**
- Disgust: *656 images*
- Sad: **2,236 images**
- Surprise: 1,474 images

This breakdown provides a detailed view of the distribution of images across different emotions for each folder in the MMAFEDB dataset.

# Model Development
..

## Model Metrics and Hyper Parameters
In this notebook, an in-depth evaluation of emotion recognition models trained on the cleaned dataset is conducted. The analysis begins with importing modules and defining a function to identify the highest metrics, such as accuracy, loss, val_accuracy, and val_loss, in the training history of each model. The model with the highest val_accuracy, identified as 'model_2_best_sgd_rgb_128,' is selected for further scrutiny. A ranking function is implemented, creating an array that ranks models based on val_accuracy. The top 10 best and 5 worst models are printed, offering a quick overview of the models' relative performance. 

Subsequently, a function plots different models using various optimizers and batch sizes, comparing the results for both rgb and grayscale color modes. This exploration aims to discern any noticeable effects of using either color mode, considering the dataset's predominant grayscale images.

To gain a nuanced understanding of the best model's performance, a dataframe is created using the model training history, capturing metrics (accuracy, loss, val_accuracy, val_loss) across epochs. A histogram illustrates the distribution of performance metrics, emphasizing the highest cumulative bin. A heatmap of val_accuracy vs. epoch provides insights into how quickly models learn, with brighter colors indicating quicker learning.

Further, the dataframe is leveraged to find max and min values grouped by model, offering a detailed snapshot of the training history. The maximum values, sorted by val_accuracy in descending order, are presented in a dataframe. Among all models, the lowest and highest values are identified, offering insights into the overall performance spectrum. Additionally, mean, median, and boxplots are calculated and visualized to provide an overview of metric distributions.

The analysis then shifts focus to the best-performing model, probing its training and validation accuracy, loss, and confusion matrix. The confusion matrix highlights correct predictions on the diagonal and exposes challenges in predicting certain emotions. The classification report offers detailed metrics, revealing f1-scores, precision, recall, and support for each emotion label. 

A Micro- and Macro-averaging ROC curve analysis follows, illustrating the trade-off between True Positive and False Positive rates with corresponding AUC values. Lastly, the best model undergoes evaluation using the test set with batch_size = 1, yielding insights into both training and validation accuracies. This multifaceted evaluation provides a comprehensive understanding of the trained emotion recognition models, shedding light on their strengths, challenges, and overall performance characteristics.

![Model 1 Image](/images/model_diagrams.png)

Results from iteratively training 4 different models that can be found in /models:
- model_1_concept - Model as initially described in the concept phase
- model_2_best - Model that was found in experimenting with different architectures and reading literature
- model_3_best_nodropout - The result of discarding dropout from model_2
- model_4_best_nodroupout_nobatchn - The result of discarding dropout and batch normalization from model_2

Using three channels (RGB) for image input the following results were obtained:
![RGB Model Training Results](/images/training_results_rgb.jpg)

Using one channel (GRAYSCALE) for image input the following results were obtained:
![GRAYSCALE Model Training Results](/images/training_results_grayscale.jpg)

As it can be seen the RGB training results are slightly better eventough a large part of the training data is grayscale images. The model seems to learn aspects from the RGB data that are not available with only one channel.

## Best performing model
The best model was selected based on the final maximum training validation accuracy (val_accuracy).

The model file with the highest val_accuracy is: ../models/training/history/model_2_best_sgd_rgb_128_augment_history.json

The mean of all models max achieved val_accuracy is: 0.5652485278745493

The highest val_accuracy (model_2_best_sgd_rgb_128) is: 0.5928241014480591

![metrics model_2_best_sgd_rgb_128](/images/best_model_metrics.png)
The best model performed ~2.8% better than the mean val_accuracy training validation.

However if we use batchsize of one when evaluating we see that the best model performs better.
    Final Results by evaluating model (batchsize = 1):
    Train accuracy = 83.28%
    Validation accuracy = 63.52%

![metrics model_2_best_sgd_rgb_128 balanced](/images/best_model_metrics_balanced.png)

![metrics model_2_best_sgd_rgb_128 balanced 2](/images/best_model_metrics_balanced-2.png)

### ROC Curve
<!-- ![roc model_2_best_sgd_rgb_128](/images/roc_curve_best_model.png)
![roc model_2_best_sgd_rgb_128 balanced](/images/roc_curve_best_model_balanced.png)
![roc model_2_best_sgd_rgb_128 balanced 2](/images/roc_curve_best_model_balanced-2.png) -->
![ROC Curve for the best model unbalanced & balanced emotions](/images/roc_curve_all.png)

**Label Encodings:**
- 0: anger
- 1: disgust
- 2: fear
- 3: happiness
- 4: sadness
- 5: surprise
- 6: neutral

### Confusion Matrix
<!-- ![confusion_matrix model_2_best_sgd_rgb_128](/images/confusion_matrix_best_model.png)
![confusion_matrix model_2_best_sgd_rgb_128 balanced](/images/confusion_matrix_best_model_balanced_emotions.png)
![confusion_matrix model_2_best_sgd_rgb_128 balanced 2](/images/confusion_matrix_best_model_balanced-2_emotions.png) -->
![Confusion Matrixes for the best model unbalanced & balanced emotions](/images/confusion_matrix_all.png)

# Front-end of Docker API
### See more information in docker-api/README.md
The retrained model uses the emotion images of the balanced-2 dataset for its predictions.
- The user selectes a file from their device
- The user clicks predict
- The prediction emotion and class probabilities are displayed.
![docker-api front-end](/images/docker_api.png)

# How to get started
## Dependencies for using the Jupyter Notebooks - Note if no CUDA GPU is present it will default to use CPU (slow).
Create a new virtual python environment for the notebooks.

`python3 -m venv venv`

Activate the environment (Linux)

`source venv/bin/activate`

Install the dependencies

`pip3 install -r requirements.txt`

## API usage using pre-build docker image
### Pull the latest build image
`docker pull deusnexus/emotion_classification:latest`
### Run the container
`docker run --name emotion_prediction_container -p 8000:8000 emotion_prediction_fastapi:latest`
### Open the API on localhost
`http://127.0.0.1:8000`

# Reflection
What I learned ...
What challenges occured ...
Installing CUDA drivers for NVIDIA Graphic Card
What recommendations are suggested ...
What could be improved ...

# Conclusion
Did we achieve the goal etc and how?

# Disclaimer
The developed application is licensed under the GNU General Public License.