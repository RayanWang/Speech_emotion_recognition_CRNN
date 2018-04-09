from optparse import OptionParser
from dataparser import DataParser
from cnnlstmmodel import CnnLstmModel
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dataset", dest="db_type", default="berlin")
    parser.add_option("-p", "--dataset_path", dest="path", default="")

    (options, args) = parser.parse_args(sys.argv)

    db_type = options.db_type
    path = options.path

    print("Loading data from " + db_type + " dataset...")
    if db_type not in ('berlin'):
        sys.exit("Dataset not registered. Please create a method to read it")

    db = DataParser(path, db_type)

    # k_folds = len(db.test_sets)
    # splits = zip(db.train_sets, db.test_sets)

    callback_list = [
        EarlyStopping(
            monitor='acc',
            patience=1,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            filepath='cnnlstm_model.h5',
            monitor='val_loss',
            save_best_only='True'
        )
    ]

    train_input_list = [[], [], [], []]
    train_labels = []
    train_inputs = []
    test_input_list = [[], [], [], []]
    test_labels = []
    test_inputs = []

    # Create model
    crnn_model = CnnLstmModel()

    # prepare training data
    for i in range(0, len(db.train)):
        for j in range(0, len(train_input_list)):
            train_input_list[j] += db.data_sets[db.train[i]][j]

        data_size = len(db.data_sets[db.train[i]][0])
        for k in range(0, data_size):
            train_labels.append(db.targets[db.train[i]])

    # transform data to numpy array format
    for i in range(0, len(train_input_list)):
        training_data = np.asarray(train_input_list[i])[:, :, :, np.newaxis]
        print(training_data.shape)
        train_inputs.append(training_data)
    train_labels = np.asarray(train_labels)
    train_labels = to_categorical(train_labels)

    # prepare testing data
    for i in range(0, len(db.test)):
        for j in range(0, len(test_input_list)):
            test_input_list[j] += db.data_sets[db.test[i]][j]

        data_size = len(db.data_sets[db.test[i]][0])
        for k in range(0, data_size):
            test_labels.append(db.targets[db.test[i]])

    # transform data to numpy array format
    for i in range(0, len(test_input_list)):
        testing_data = np.asarray(test_input_list[i])[:, :, :, np.newaxis]
        print(testing_data.shape)
        test_inputs.append(testing_data)
    test_labels = np.asarray(test_labels)
    test_labels = to_categorical(test_labels)

    # Compile model
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
    crnn_model.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # Fit the model, feeding the validation set
    history = crnn_model.model.fit(train_inputs, train_labels, epochs=30, batch_size=64,
                                   callbacks=callback_list, validation_data=(test_inputs, test_labels),
                                   verbose=1)

    history_dict = history.history
    epochs = range(1, len(history_dict['acc']) + 1)

    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    plt.clf()

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

