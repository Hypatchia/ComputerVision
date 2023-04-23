from keras.models import Model ,load_model
from keras import Input
from keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model






# Build an AutoEncoder 

# Number of Input neurons & Bottleneck
n_inputs = X_train.shape[1]

n_bottleneck = (n_inputs // 3)


# Input Layer

Input_Layer = Input(shape=(n_inputs,), name = 'Input-Layer')
# Encoder Layer 
Enc = Dense(units = n_inputs, name = 'Encoder-Layer')(Input_Layer)
Batch_Normalization = BatchNormalization(name = 'Encoder-Layer-Normalization')(Enc)
L_ReLU = LeakyReLU(name = 'Encoder-Layer-Activation')(Batch_Normalization)

# Bottleneck Layer
BottleNeck = Dense (units = n_bottleneck , name = 'Bottleneck-Layer')(L_ReLU)

# Decoder Layer 
Dec = Dense(units = n_inputs, name ='Decoder-Layer')(BottleNeck)
D_Batch_Normalization = BatchNormalization(name = 'Decoder-Layer-Normalization')(Dec)
DL_ReLU = LeakyReLU(name ='Decoder-Layer-Activation')(D_Batch_Normalization)

# Output Layer 
Output_Layer = Dense(units = n_inputs , activation ='linear' , name = 'Output-layer')(DL_ReLU)


