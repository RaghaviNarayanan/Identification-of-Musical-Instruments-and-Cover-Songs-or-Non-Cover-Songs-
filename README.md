# Identification-of-Musical-Instruments-and-Cover-Songs-or-Non-Cover-Songs-
This project focuses on two important problems in Music Information Retrieval: identifying musical instruments from audio and detecting whether two songs are cover versions of each other. The system is designed using digital signal processing techniques combined with machine learning models to analyze and understand audio data.

The project flow begins with audio data collection. Musical audio files are taken from standard datasets. For musical instrument identification, the IRMAS dataset is used, which contains short audio clips of different instruments. For cover song detection, the Cover80 dataset is used, which contains multiple versions of the same songs performed differently. These datasets provide real-world music samples for reliable evaluation.

Once the audio files are collected, they go through a preprocessing stage. Each audio file is converted to a standard sampling rate and transformed into a single-channel signal. A fixed-length segment is extracted from every audio clip so that all samples have the same duration. The audio signals are also normalized to reduce the effect of loudness differences. This preprocessing ensures consistency and improves feature quality.

After preprocessing, important audio features are extracted from each signal. These features describe different characteristics of sound such as timbre, pitch, and spectral behavior. Mel Frequency Cepstral Coefficients capture the texture of sound, chroma features represent harmonic content, and spectral features describe brightness and frequency distribution. These features convert raw audio into numerical form that machine learning models can understand.

For musical instrument identification, the extracted features are directly passed to different machine learning models. Models such as Decision Tree, Random Forest, XGBoost, and Neural Networks are trained to classify which instrument is present in the audio. The models learn patterns in the feature values that are unique to each instrument type.

For cover song detection, the flow is slightly different. Audio files are paired together, and each pair is labeled as either a cover pair or a non-cover pair. Feature vectors from both audio files are combined along with their differences to form a single input. These inputs are then given to classification models that decide whether the two songs are versions of the same composition.

During training, different feature transformation methods such as PCA and SelectKBest are tested to study their effect on performance. The models are evaluated using accuracy, F1-score, and confusion matrices. The results show that ensemble-based models perform best when the complete feature set is used.

Overall, the project flow starts with audio data acquisition, followed by preprocessing and feature extraction, then moves to machine learning model training and testing, and finally ends with classification results for instrument recognition and cover song detection. This flow demonstrates how digital signal processing and machine learning can be combined to solve real-world music analysis problems.
