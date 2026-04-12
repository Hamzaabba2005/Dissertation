from sklearn.model_selection import GroupShuffleSplit

def patient_level_split(samples, val_size=0.2, seed=42):
    groups = [s["patient_id"] for s in samples]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed
    )

    train_idx, val_idx = next(gss.split(samples, groups=groups))

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    return train_samples, val_samples

