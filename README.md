# IU INTERNATIONAL UNIVERSITY OF APPLIED SCIENCES

### Project: Artificial Intelligence (DLBDSEAIS02)
Utilizing TensorFlow as the primary framework, the project focuses on emotion detection from facial expressions using a Convolutional Neural Network (CNN). TensorFlow's deployment capabilities, including TensorFlow Serving and TensorFlow Lite, ensure seamless transition to production. The workflow involves a pre-processing pipeline, transforming images to a 48x48 dimension from the mma-facial-expression dataset. The CNN architecture comprises convolution layers, max pooling, ReLU activation, dense layers, and a fully connected layer with Softmax activation. Model testing involves metrics like accuracy, recall, and precision, with the F1-score for multi-classification. Deployment is done via FastAPI and Docker, enabling local or cloud-based execution.

## Purpose
Build an AI tool for the marketing department to analyze emotional responses to company ads using facial expressions in images. Develop a system to categorize emotions (e.g., joy, anger, fear) from an image dataset. Choose relevant technologies for components and communication. The goal is to identify at least three emotional states through facial expressions.

# Development Planning - UML Schemas
## UML Diagram
Diagram - Design of Convolutional Neural Network to classify emotions and serve using API
![Design of Convolutional Neural Network to classify emotions and serve using API](/images/UML.jpg)

## Dataset Overview
Source
Folders, Image Data, Count/Size, Balanced or Unbalanced?

# Model Development
..

## Model Metrics and Hyper Parameters
Accurary, Recall, Precision ...
optimizer, loss function, etc.

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


### ROC Curve
![roc model_2_best_sgd_rgb_128](/images/roc_curve_best_model.png)

### Confusion Matrix
![confusion_matrix model_2_best_sgd_rgb_128](/images/confusion_matrix_best_model.png)

**Label Encodings:**
- 0: anger
- 1: disgust
- 2: fear
- 3: happiness
- 4: sadness
- 5: surprise
- 6: neutral

# How to get started
## Dependencies
...
...

## Installation instructions
...

## API usage
...

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