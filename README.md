
# Решение: определение ориентации текста (0° / 180°)

## Задача
Бинарная классификация: определить, повёрнут ли текст в боксе на 180°.
Метрика: 1 - Brier Score.

## Данные
Обучающие данные не предоставлялись. Был сгенерирован синтетический
датасет с помощью утилиты `text2image` из Tesseract OCR.

- 33 шрифта: Arial, Helvetica, Helvetica Neue, Verdana, Tahoma,
  Trebuchet MS, Geneva, Avenir, Optima, Times New Roman, Georgia,
  Baskerville, Didot, Palatino, Rockwell, Courier New, Chalkboard,
  Chalkduster, Comic Sans MS, Papyrus, Copperplate и другие.
- Размеры шрифта: от 24 до 60 pt.
- Тексты: бренды (Kinder, Michael Kors), магазинные термины
  (Open, Sale, Discount), названия городов (Moscow, London, Paris),
  короткие фразы из 2 слов.
- Разрешение: 72 DPI, картинки приводятся к 64×64 пикселей.

Объём: 2000 нормальных изображений + 2000 перевёрнутых =
4000 изображений (баланс классов 1:1).

## Признаки
Из каждого бокса извлекаются:
- HOG (Histogram of Oriented Gradients):
  orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1).
- Статистики яркости: mean, std.
- Проекции по строкам и столбцам: max, mean, std.

Итоговый вектор признаков: 130 чисел.

## Модель
LogisticRegression (scikit-learn, max_iter=1000, random_state=42).
Выбор обоснован требованием к компактности и производительности:
модель занимает несколько десятков KB и обрабатывает
~2000 изображений/сек на CPU.

## Валидация
Holdout 80/20 со стратификацией:
train_test_split(test_size=0.2, stratify=y, random_state=42).

## Результаты
Brier Score: 0.0256
Метрика 1 - Brier Score: 0.9744
Accuracy: 0.9688

## Использованные open-source библиотеки
- Tesseract OCR (text2image) — Apache 2.0
- OpenCV — Apache 2.0
- scikit-image (HOG) — BSD-3
- scikit-learn (LogisticRegression) — BSD-3
- NumPy, Pandas — BSD-3

## Воспроизводимость
Все random seed зафиксированы (random_state=42).
Pipeline запускается в 3 шага:
1. python generate_dataset.py — генерация датасета.
2. python train_model.py — обучение модели -> model.pkl.
3. python predict.py — генерация submission.csv.



