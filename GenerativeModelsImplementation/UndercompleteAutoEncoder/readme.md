### Task :  Python | TensorFlow Functional API | Numpy | UndercompleteAE | Tabular Data Recosntruction 
* Objectif is to train an UndercompleteAutoEncoder to reconstruct a dataset while learning about AEs' training and Architecture .
#### AutoEncoders, Unsupervised , Semi-Supervised

* AE take an input vector , encode it into a latent space , decode it for an output 
    * AE mainly recreate inputs with minimal loss 
* Two main types :
    * Undercomplete : Middle layers with less neuronal dimension than input and output : bottleneck
    * Overcomplete : Middle layers with higher dimension than input and output



##### About The Dataset :
* Dataset : https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
* Technology : 
   * Python 
   * Tensorflow 
   * keras Functional API
* Content :
    * This dataset contains about 10 years of daily weather observations from many locations across Australia.

* Context :
    * The Objective is:
    * To build and train an Uncomplete AutoEncoder 
    * The Challenge is an Unsupervised-SemiSupervised Problem 
    * 17 features from the dataset were taken as a Feature Space
    * The features were compressed into a BottleNeck of 1/3 the input size and decoded
    * A decoded output is of size 17 
    * The Validation Loss was minimal
    
* Architecture of the AUtoEncoder :
<img src="https://github.com/Hypatchia/GenerativeModeling/blob/main/UncompleteAutoEncoder/AutoEncoder.png" height="40%" width="40%" >

  
  
  
* Source & Credit :
    * Observations were drawn from numerous weather stations. The daily observations are available from http://www.bom.gov.au/climate/data.


    * Data source: http://www.bom.gov.au/climate/dwo/ and http://www.bom.gov.au/climate/data.

    * Copyright Commonwealth of Australia 2010, Bureau of Meteorology.
