from tensorflow import keras
# Encoder Layers 
recurrent_encoder = keras.models.Sequential([
 keras.layers.LSTM(100, return_sequences=True, input_shape=[None, 28]),
 keras.layers.LSTM(30)
])


# Decoder Layers
recurrent_decoder = keras.models.Sequential([
 keras.layers.RepeatVector(28, input_shape=[30]),
 keras.layers.LSTM(100, return_sequences=True),
 keras.layers.TimeDistributed(keras.layers.Dense(28, activation="sigmoid"))
])

# Reccurent AutoEnoder Model
recurrent_ae = keras.models.Sequential([recurrent_encoder, recurrent_decoder])