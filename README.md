# Speech_emotion_recognition using CNN and LSTM
A deep learning application for speech emotion recognition.

Language: Python 3.6

# Dependencies

- [keras](https://github.com/keras-team/keras) for building/training the CNN + LSTM network
- [librosa](https://github.com/RayanWang/librosa) for doing STFT

# Datasets

- [Berlin speech dataset](http://emodb.bilderbar.info/download/)

# Usage

Long option | Option | Description
----------- | ------ | -----------
--dataset | -d | dataset type
--dataset_path | -p | dataset path

Example:

    python emotionrecognition.py -d 'berlin' -p [berlin db path]

# References

- Wootaek Lim, Daeyoung Jang and Taejin Lee, Speech Emotion Recognition using Convolutional and Recurrent Neural Networks, Audio and Acoustics Research Section, ETRI, Daejeon, Korea
