from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Permute, LSTM, concatenate, Dense
from keras.layers.core import Reshape
from keras.models import Model


class CnnLstmModel:

    def __init__(self, input_shape=(128, 128, 1), nb_classes=7):
        dataset = []
        layers = []

        for i in range(0, 4):
            dataset.append(Input(shape=input_shape))

        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)
        # number of LSTM neural units
        nb_lstm_units = 1024

        for i in range(1, 5):
            suffix = '_' + repr(i) + '_'
            x = Convolution2D(nb_filters, kernel_size, padding='same', name='conv' + suffix + '1',
                                  activation='relu')(dataset[i - 1])
            x = MaxPooling2D(pool_size, name='maxpool' + suffix + '1')(x)
            x = Dropout(0.25, name='dropout' + suffix + '1')(x)
            x = Convolution2D(nb_filters, kernel_size, padding='same', name='conv' + suffix + '2',
                                  activation='relu')(x)
            x = MaxPooling2D(pool_size, name='maxpool' + suffix + '2')(x)
            x = Dropout(0.25, name='dropout' + suffix + '2')(x)
            x = Reshape((-1, nb_filters))(x)
            x = Permute((2, 1))(x)
            x = LSTM(nb_lstm_units, name='LSTM' + suffix + '1', dropout=0.2, recurrent_dropout=0.2,
                         return_sequences=True)(x)
            x = LSTM(nb_lstm_units, name='LSTM' + suffix + '2', dropout=0.2, recurrent_dropout=0.2)(x)
            layers.append(x)

        x = concatenate(layers, name='concat', axis=-1)
        main_output = Dense(nb_classes, activation='softmax')(x)
        self.model = Model(inputs=dataset, outputs=main_output)
        self.model.summary()
