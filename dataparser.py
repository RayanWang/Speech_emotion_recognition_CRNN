import librosa
import os


class DataParser:

    # Dataset object is composed of:
    # data
    # targets
    # train and test sets
    # classes dictionary to map classes to numbers

    def __init__(self, path, type):
        if type == "berlin":
            self.classes = {0: 'W', 1: 'L', 2: 'E', 3: 'A', 4: 'F', 5: 'T', 6: 'N'}
            self.get_berlin_dataset(path)

    def get_berlin_dataset(self, path):
        # males = ['03', '10', '11', '12', '15']
        # females = ['08', '09', '13', '14', '16']
        classes = {v: k for k, v in self.classes.items()}

        self.targets = []
        self.data_sets = []
        self.train = []
        self.test = []

        hop_size = 128
        speaker_test = ['03', '08']
        i = 0
        for audio in os.listdir(path):
            audio_path = os.path.join(path, audio)
            y, sr = librosa.load(audio_path, sr=16000, mono=True)  # load audio data
            src = librosa.stft(y, n_fft=256, hop_length=hop_size, center=False)

            bucket_0 = []
            bucket_1 = []
            bucket_2 = []
            bucket_3 = []
            for j in range(0, src.shape[1], 8):
                if (j + 24) >= src.shape[1] or (j + 24 + hop_size) >= src.shape[1]:
                    break
                bucket_0.append(src[0: hop_size, j: j + hop_size])
                bucket_1.append(src[0: hop_size, j + 8: j + 8 + hop_size])
                bucket_2.append(src[0: hop_size, j + 16: j + 16 + hop_size])
                bucket_3.append(src[0: hop_size, j + 24: j + 24 + hop_size])

            self.data_sets.append([bucket_0, bucket_1, bucket_2, bucket_3])
            self.targets.append(classes[audio[5]])

            if audio[:2] in speaker_test:
                self.test.append(i)
            else:
                self.train.append(i)
            i = i + 1
