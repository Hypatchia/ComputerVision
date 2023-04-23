import pandas as pd
from tensorflow import keras
print('pandas: %s' % pd.__version__) 
print('tf keras : %s' % keras.__version__)

from keras.models import Model ,load_model
from keras import Input
from keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model


import sklearn 
print('sklearn: %s' % sklearn.__version__) 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import matplotlib 
import matplotlib .pyplot as plt
print('matplotlib: %s' % matplotlib.__version__) 

import graphviz
print('graphviz: %s' % graphviz.__version__)




# Data Pre-Processing
# Display k columns in pandas 
pd.options.display.max_columns = 30

# Read australian wheather data 
df = pd.read_csv('weatherAUS.csv',encoding = 'utf-8')

# Drop rows with missing values

df = df.dropna(axis = 0)

# Create flag for RainToday column 

df['RainTodayFlag'] = df['RainToday'].apply(lambda x:1 if x=='Yes' else 0 )



# Data Processing
# Feature Selection

X = df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 
      'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',  
      'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainTodayFlag']]


# MinMaxScaling

Scaler = MinMaxScaler()
X_scaled = Scaler.fit_transform(X)



# Train & Test Sample

X_train , X_test = train_test_split(X_scaled , test_size = 0.25, random_state = 0 )




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




# The AutoEncoder Model

model = Model(inputs = Input_Layer , outputs = Output_Layer , name ='AutoEncoder-Model')
# Compile Model
model.compile(optimizer = 'adam' ,loss = 'mse')
# Model Summary 
print(model.summary()) 
# AE Diagram 
plot_model(model,show_shapes = True ,dpi =300 )



# Train on k-epochs
k_epochs = 15

history = model.fit(X_train , X_train,epochs = k_epochs, batch_size = 16, verbose = 1, validation_data = (X_test, X_test))
# Plot a loss chart
fig, ax = plt.subplots(figsize=(16,9), dpi=300)
plt.title(label='Model Loss by Epoch', loc='center')

ax.plot(history.history['loss'], label='Training Data', color='black')
ax.plot(history.history['val_loss'], label='Test Data', color='red')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.legend()

plt.show()





# Encoder Alone
The_Encoder = Model(inputs = Input_Layer, outputs = BottleNeck)

# Compile Encoder 

The_Encoder.compile(optimizer = 'adam' , loss  = 'mse')

# Save 

The_Encoder.save('Encoder.h5')

# Decoder Alone 
The_Decoder = Model (inputs = BottleNeck , outputs = Output_Layer )

# Compile Decoder 
The_Decoder.compile(optimizer = 'adam' , loss  = 'mse')

# Save 

The_Encoder.save('Decoder.h5')

# Save AutoEncoder 
model.save('AutoEncoder.h5')



# Plot Diagrams and Save them


# Encoder
plot_model(The_Encoder, 'Encoder.png', show_shapes=True, dpi=300)

#Decoder
plot_model(The_Decoder, 'Decoder.png', show_shapes=True, dpi=300)