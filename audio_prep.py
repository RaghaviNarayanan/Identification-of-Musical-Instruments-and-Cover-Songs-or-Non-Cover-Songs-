import os, glob, librosa, numpy as np
import pandas as pd

def extract_features(file_path, n_mfcc=20):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if len(y) < 2048:
        print(f"Skipping {file_path}: too short")
        return None

    n_fft = min(2048, len(y))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    

    # Spectral
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft))
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99, n_fft=n_fft))

    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft), axis=1)

    # Combine all
    return np.hstack([
        [mfcc_mean,
        mfcc_std,]
        
        [spec_centroid, spec_bandwidth, zcr, rolloff],
        chroma,
    ])

def dataset_to_csv(folder, out_csv, n_mfcc=20):
    data = []
    for label in os.listdir(folder):
        class_dir = os.path.join(folder, label)
        if not os.path.isdir(class_dir):
            continue
        files = glob.glob(os.path.join(class_dir, "*.wav"))
        for f in files:
            try:
                features = extract_features(f, n_mfcc)
                if features is None:
                    continue
                row = features.tolist()
                row.append(label)
                data.append(row)
            except Exception as e:
                print(f"Error with {f}: {e}")

    columns = (
        [f"mfcc_mean_{i+1}" for i in range(n_mfcc)] +
        [f"mfcc_std_{i+1}" for i in range(n_mfcc)] +
        ["spec_centroid", "spec_bandwidth", "zcr", "rolloff"] +
        [f"chroma_{i+1}" for i in range(12)] +
        ["label"]
    )

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved {out_csv} with {len(df)} samples and {len(columns)-1} features.")

if __name__ == "__main__":
    dataset_to_csv(
        folder=r"D:\download-cdrive\IRMAS-TrainingData\IRMAS-TrainingData",
        out_csv="irmas_train.csv",
        n_mfcc=20
    )
