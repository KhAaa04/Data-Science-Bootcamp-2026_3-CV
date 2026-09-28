import os
import glob
import cv2
import numpy as np
import joblib
import pandas as pd
from skimage.feature import hog
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    hog_features = hog(
        img, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), visualize=False
    )
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    row_proj = np.sum(img, axis=1) / 255.0
    col_proj = np.sum(img, axis=0) / 255.0
    row_stats = [np.max(row_proj), np.mean(row_proj), np.std(row_proj)]
    col_stats = [np.max(col_proj), np.mean(col_proj), np.std(col_proj)]
    return np.concatenate([
        hog_features,
        [mean_intensity, std_intensity],
        row_stats,
        col_stats
    ])


def main():
    MODEL_PATH = "/Users/anna/testpy/model.pkl"
    TEST_DIR = "/Users/anna/testpy/images"
    OUTPUT_CSV = "/Users/anna/testpy/submission.csv"

    # загрзка модели
    model = joblib.load(MODEL_PATH)

    # тестовые файлы
    test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*")))
    print(f"Найдено тестовых файлов: {len(test_files)}")
    if len(test_files) == 0:
        raise SystemExit("ОШИБКА: тестовые файлы не найдены! Проверьте TEST_DIR")

    # предсказание
    results = []
    for idx, path in enumerate(test_files):
        filename = os.path.splitext(os.path.basename(path))[0]
        feats = extract_features(path)
        if feats is None:
            prob = 0.5
        else:
            prob = model.predict_proba([feats])[0][1]
        results.append({"image_id": filename, "p_180": prob})


    # сохранение
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nСохранено: {OUTPUT_CSV}")
    print(f"Строк: {len(df)}")
    print(f"\nПервые 5 строк:")
    print(df.head())
    print(f"\nРаспределение p_180:")
    print(df["p_180"].describe())


if __name__ == "__main__":
    main()
