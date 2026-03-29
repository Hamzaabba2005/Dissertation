import os

def pick_images_dir(base_path, df):
    candidates = []

    for name in os.listdir(base_path):
        p = os.path.join(base_path, name)
        if os.path.isdir(p):
            candidates.append(p)

    test_names = []
    for i in range(min(20, len(df))):
        test_names.append(str(df.iloc[i]["Left-Fundus"]).strip())
        test_names.append(str(df.iloc[i]["Right-Fundus"]).strip())

    best = None
    best_hits = -1

    for cand in candidates:
        hits = sum(os.path.exists(os.path.join(cand, nm)) for nm in test_names)
        if hits > best_hits:
            best = cand
            best_hits = hits

    return best, best_hits