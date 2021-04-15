import os
import numpy as np
from sklearn.preprocessing import scale
import librosa
import matplotlib.pyplot as plt

def save_melspectrogram(directory_path, file_name, sampling_rate=44100):
    """ Will save spectogram into current directory"""

    path_to_file = os.path.join(directory_path, file_name)
    data, sr = librosa.load(path_to_file, sr=sampling_rate, mono=True)
    data = scale(data)

    melspec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
    # Convert to log scale (dB) using the peak power (max) as reference
    # per suggestion from Librbosa: https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    librosa.display.specshow(log_melspec, sr=sr)

    # create saving directory
    directory = './melspectrograms'
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory + '/' + file_name.strip('.wav') + '.png')

def _train_test_split(filenames, train_pct):
    """Create train and test splits for ESC-50 data"""
    random.seed(2018)
    n_files = len(filenames)
    n_train = int(n_files * train_pct)
    train = np.random.choice(n_files, n_train, replace=False)

    # split on training indices
    training_idx = np.isin(range(n_files), train)
    training_set = np.array(filenames)[training_idx]
    testing_set = np.array(filenames)[~training_idx]
    print('\tfiles in training set: {}, files in testing set: {}'.format(len(training_set), len(testing_set)))

    return {'training': training_set, 'testing': testing_set}

# Load meta data for audio files
meta_data = pd.read_csv('meta/esc50.csv')

labs = meta_data.category
unique_labels = labs.unique()

for label in unique_labels:
    print("Proccesing {} audio files".format(label))
    current_label_meta_data = meta_data[meta_data.category == label]
    datasets = _train_test_split(current_label_meta_data.filename, train_pct=0.8)
    for dataset_split, audio_files in datasets.items():
        for filename in audio_files:
            directory_path = 'audio/'
            save_melspectrogram(directory_path, filename, dataset_split, label, sampling_rate=44100)
    