# Music Genre Classification

In this project I try to explore music audio features and train a DL model on the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) which contains 30 second music `.wav` file over 10 genres. This work is based on the teaching of Valerio Velardo (Shoutout to his YouTube channel by the same name). In the `IntroToAudioFeatures` folder, `preprocessing.py` file contains the basic operations on a audio `.wav` file using the `librosa` library. I compute the `FFT, STFT, and MFCCs` of a sample audio file.

The `GenreClassifier` contains the 2 models (Fully connected neural network and an LSTM model) which perform the classification task over 10 categories. Since we only have 100 data samples for each genre. We use data augmentation by by chopping each track into different samples. 
``` 
Number of segments per track = 10
Hoplength for Fourier Transforms = 512
Total Duration = 30 secs
Segment Duration = 3 secs
Sample rate = 22050
Num of MFCCs = 13
```
The overall dataset shape is 
```
X = (num_genres * num_songs_in_genre * num_segments x num_hops_per_segment x 13)
Y = (num_genres * num_songs_in_genre * num_segments x 1)
```

For the FNN network, the number of hops per segments are flattened and passed into the network where as for the case of an LSTM, the hops per segments as kept as a temporal dimension for the RNN.

FNN Test Accuracy = 54%
LSTM Test Accuracy = 65%
