### Task :  Python | TensorFlow | Numpy | Convolutional Neural Nets | Image Classifciation

* Build a Convolutional Neural Network to Classify Images into Defected and non Defected on a manufacturing production line.
* This is a Project that I created to use as teaching material of CNNs during my experience as an AI intsructor in my previous Company.
* This is an end to end guided project for learning about Binary Image Classification using CNNs in TensorFlow.



### Classifying Products to Defected VS Not Defected (OK) based on their front image :
* **Dataset** : Kaggle Dataset : https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product
* **Data** : The data contains 7348 grayscale images for a manufacturing product, 300x300 pixel images , augmentation results included.

* Business Understanding: What is the Business Objective ?
  * The objective is to detect whether a manufactured product is defected or not
* Use Case:
  * The production company wants to eliminate the defected products so the ones classifid as defected will not be sent to shipping .
* Understanding our ML Pipeline:
  * **Input** : 300 by 300 grayscale images with 1 color layer 
  * **output** : Classes of the Images , 0 as defected and 1 as Safe or OK
* Framing the Problem :
  * The problem here is an Image Classification task using a CNN
* Designing the System
  * The Value of the output is 0 or 1 So the goal is to predict labels for images "Ok_front" and "def_front'
  * We will be having a Binary classification Task.

 * Download Notebook 
 * Download the Dataset 
 * Run the notebook while followig the guided steps .

  ![](https://img.shields.io/badge/Python3.8-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=Blue)
  ![](https://img.shields.io/badge/Tensorflow-informational?style=flat&logo=<LOGO_NAME>&logoColor=white&color=orange)
     
