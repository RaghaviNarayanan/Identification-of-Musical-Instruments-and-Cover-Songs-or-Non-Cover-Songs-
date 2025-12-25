import os, glob, librosa, numpy as np
import pandas as pd

def extract_features(file_path, n_mfcc=20):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if len(y) < 2048:
        print(f"Skipping {file_path}: too short")
        return None

    n_fft = min(2048, len(y))

  
    

    # Spectral
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft))
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft))
    
    return np.hstack(
        [spec_centroid ,spec_bandwidth]
        )

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
        
        ["spec_centroid", "spec_bandwidth"] +
        
        ["label"]
    )

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved {out_csv} with {len(df)} samples and {len(columns)-1} features.")

if __name__ == "__main__":
    dataset_to_csv(
        folder=r"D:\download-cdrive\IRMAS-TrainingData\IRMAS-TrainingData",
        out_csv="irmas_train_spec.csv",
        n_mfcc=20
    )
