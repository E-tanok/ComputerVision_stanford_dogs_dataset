# The project

The aim of this project is to put in place methods that make it possible to identify the race from an incoming dog image. This project is broken down into several stages whose sequencing is illustrated on the "ROADMAP" below:
![alt text](https://github.com/E-tanok/projects_pictures/blob/master/ComputerVision/stanford_dogs_dataset/ROADMAP.png)
*The "ROADMAP" breaks down into three main stages: Preprocessing, Learning and Evaluation*

##  Preprocessing

### part_1_renaming_folders_and_files :

In this program, we restructure the folder names associated with dog races as well as the names of dog pictures. The goal is to have a more appropriate and understandable data structure


### part_2_datasets_building  :

This program builds the train, validation and test perimeters that will be used to build the learning models

      - **Classical approach**: A dictionary containing the train and test perimeters is built. The SIFT features associated with each image of the train and test perimeters are also constructed. Two dataframes (one for the train data, another for the test data) are then saved for the next step. The granularity of the data is at the level of the SIFT features (a row of datasets corresponds to a feature detected on an image, a given image having N features SIFT)

      - **CNN approach**: Another dictionary, containing the train perimeters, validation and test, is built for the CNN approach

## Learning (classical approach)

### part_3_classical_approach_SIFT_features_analysis :

In this program, we perform analyzes on SIFT features generated for images and dogs races

### part_4_classical_approach_clustering_KM :

In this program, we build different datasets of bag of visual words. To do this, we implement different KMeans clustering: 50, 100, 200,300 and 500 clusters. The bag of visual words datasets are then obtained in two steps:

      - Aggregation of the SIFT descriptor columns of each image-level feature: For each descriptor (descriptor) on all feature lines attached to the image. The granularity of the data becomes that of the images.

      - Standardization, for each image, of the sums of each descriptor relative to the sum of all the descriptors

A bag of visual words dataset is built for each previously calculated type of clustering: each bag of visual words dataset is saved for the next step.

### part_5_classical_approach_bag_of_visual_words_classification :

This program implements three classification algorithms: Logistic Regression, Support Vector Classifier, and Random Forest. The goal is to predict the dogs races through the datasets of bag of visual words.

      - A first part uses, for a given algorithm, cross validation.

      - A second part uses the best parameters of the cross validation performed on the algorithm in order to build a final learning model.


## Evaluation (classical approach) :

### part_6_classical_approach_classification_iterations  :

In this program, we loop over the previous script, **part_5_classical_approach_bag_of_visual_words_classification** , in order to implement its classification algorithms with all bag of visual words datasets


##  Learning (CNN approach)

N.B: These notebooks have been set up thanks to the work done on the notebooks provided in the appendix: **Annex_1_CNN_approach_transfer_learning_over_parameters** and **Annex_2_CNN_approach_results_analyzis**

### part_7_CNN_approach_transfer_learning :

This program implements the transfer learning of the VGG16 model driven on IMAGENET.

We freeze the learned weights on the first 13 layers of convolution and we train only the last 3 layers fully connected.
Various network parameters (Batch size, learning rate, and optimizer) are varied.
The training is carried out over 100 eras. It is performed by cross-validation on the validation perimeter defined in **part_2_datasets_building** . An "Early Stopping" is set so that the training ends when the validation loss does not change after 3 epochs.

As a regularization, a Dropout layer is added after the first two layers fully connected. Each dropout layer obscures 20% of the neural signals it receives.

### part_8_CNN_approach_results_analyzis_transfer_learning :

This program analyzes the results of the **part_7_CNN_approach_transfer_learning** notebook processes

### part_9_CNN_approach_final_model :

In this program, we train a final neural network based on the best results obtained in **part_8_CNN_approach_results_analyzis_transfer_learning** .


# The flask application :

The CNN approach allowed me to build a flask application which is in another [github project](http://bit.ly/mk_cv_dogs) :

![alt text](https://github.com/E-tanok/ComputerVision_dogedex/blob/master/project_instructions/first_steps.jpg)  
