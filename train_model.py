import os
import glob
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
# извлечение признаков
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    # HOG — улавливает направление штрихов в буквах
    hog_features = hog(
        img, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), visualize=False
    )
    # Дополнительные статистики
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


# загрузка датасета
X, y = [], []

with open("/Users/anna/testpy/dataset/train.txt") as f:
    lines = f.readlines()
for idx, line in enumerate(lines):
    path, label = line.strip().split("\t")
    feats = extract_features(path)
    if feats is not None:
        X.append(feats)
        y.append(int(label))

X = np.array(X)
y = np.array(y)
print(f"Признаки: {X.shape}, Метки: {y.shape}")
print(f"Класс 0: {(y == 0).sum()}, Класс 1: {(y == 1).sum()}")

# TRAIN / VAL SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nОбучающая выборка: {X_train.shape}")
print(f"Валидационная выборка: {X_val.shape}")

# обучение
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# оценка
y_pred_proba = model.predict_proba(X_val)[:, 1]
bs = brier_score_loss(y_val, y_pred_proba)
print(f"Brier Score: {bs:.4f}")
print(f"Метрика 1 - Brier Score: {1 - bs:.4f}")
y_pred = (y_pred_proba > 0.5).astype(int) # Точность на пороге 0.5
accuracy = (y_pred == y_val).mean()
print(f"Accuracy: {accuracy:.4f}")

# сохранение
joblib.dump(model, "/Users/anna/testpy/model.pkl")
