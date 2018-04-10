from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Permute, LSTM, Dense, concatenate
from keras.layers.core import Reshape
from keras.models import Model


class CnnLstmModel:

    def __init__(self, nb_inputs=4, input_shape=(128, 128, 1), nb_classes=7):
        dataset = []
        layers = []

        for i in range(0, nb_inputs):
            dataset.append(Input(shape=input_shape))

        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)
        # number of LSTM output units
        nb_lstm_cells = 1024

        for i in range(0, len(dataset)):
            x = Convolution2D(nb_filters, kernel_size, padding='same', activation='relu')(dataset[i])
            x = MaxPooling2D(pool_size)(x)
            x = Dropout(0.25)(x)
            x = Convolution2D(nb_filters, kernel_size, padding='same', activation='relu')(x)
            x = MaxPooling2D(pool_size)(x)
            x = Dropout(0.25)(x)
            x = Reshape((-1, nb_filters))(x)
            x = Permute((2, 1))(x)
            x = LSTM(nb_lstm_cells, return_sequences=True)(x)
            x = LSTM(nb_lstm_cells)(x)
            layers.append(x)

        x = concatenate(layers)
        main_output = Dense(nb_classes, activation='softmax')(x)
        self.model = Model(inputs=dataset, outputs=main_output)
        self.model.summary()
