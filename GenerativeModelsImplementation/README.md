# GenerativeModeling




#### Convolutional Auto Encoder Architecture

  * Technology : Keras Sequential API
  * Suitable for Image generation
  * Encoder 
    * Regular CNN
      * Convolution + pooling layers 
  * Decoder
    * Reverse CNN 
      * Transpose Conv layers + pooling Layers

  * Training : Similar to a Regular Neural Network 
    * Define Model Architecture
    * Choose **Loss** and **Optimizer**
    * **Compile**
    * **Fit**
    * **Predict** 






#### Reccurent Auto Encoder Architecture

  * Technology : Keras Sequential API
  * Suitable for Text and Sequence data generation
  * Encoder 
    * Regular RNN
      * Sequence-to-vector RNN
  * Decoder
    * Reverse RNN
      * vector-to-sequence RNN

  * Training : Similar to a Regular Neural Network 
    * Define Model Architecture
    * Choose **Loss** and **Optimizer**
    * **Compile**
    * **Fit**
    * **Predict** 




