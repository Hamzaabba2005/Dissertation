import os
import numpy as np


def build_samples_existing(df, images_dir, label_cols):
    samples = []
    missing = 0
    bad_labels = 0

    for _, r in df.iterrows():
        y = r[label_cols].to_numpy(dtype=np.float32)

        # skip impossible labels
        if np.isnan(y).any():
            bad_labels += 1
            continue

        patient_id = int(r["ID"])

        eye_info = [
            ("Left-Fundus", "Left"),
            ("Right-Fundus", "Right"),
        ]

        for img_col, side in eye_info:
            fname = str(r[img_col]).strip()

            if not fname or fname.lower() == "nan":
                continue

            path = os.path.join(images_dir, fname)

            if os.path.exists(path):
                samples.append({
                    "path": path,
                    "label": y.copy(),
                    "patient_id": patient_id,
                    "filename": fname,
                    "side": side,
                })
            else:
                missing += 1

    return samples, missing, bad_labels