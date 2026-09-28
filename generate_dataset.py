import os
import glob
import random
import subprocess
import cv2
import numpy as np

# настройки
OUTPUT_DIR = "/Users/anna/testpy/dataset"
TRAIN_NORMAL = os.path.join(OUTPUT_DIR, "train/0")
TRAIN_ROTATED = os.path.join(OUTPUT_DIR, "train/1")
TMP_DIR = "/tmp"
NUM_SAMPLES = 2000

for d in [TRAIN_NORMAL, TRAIN_ROTATED]:
    os.makedirs(d, exist_ok=True)

# Готовим входной файл в /tmp
with open(os.path.join(TMP_DIR, "input_test.txt"), "w") as f:
    f.write("Hello World")

# шрифты
FONTS = [
    "Arial", "Helvetica", "Helvetica Neue", "Verdana", "Tahoma",
    "Trebuchet MS", "Geneva", "Microsoft Sans Serif",
    "Avenir", "Avenir Next", "Gill Sans", "Optima", "Seravek",
    "Times New Roman", "Times", "Georgia", "Baskerville",
    "Didot", "Charter", "Palatino", "Hoefler Text",
    "Iowan Old Style", "Bodoni 72", "Rockwell",
    "Courier New", "Courier",
    "Chalkboard", "Chalkduster", "Marker Felt",
    "Comic Sans MS", "Papyrus", "Copperplate",
    "American Typewriter",
]


def font_works(font):
    original_dir = os.getcwd()
    os.chdir(TMP_DIR)
    try:
        for f in glob.glob("_test_font*"):
            try:
                os.remove(f)
            except:
                pass

        cmd = [
            "text2image",
            "--text=input_test.txt",
            "--outputbase=_test_font",
            f"--font={font}",
            "--ptsize=30",
            "--margin=5",
            "--resolution=72",
            "--degrade_image=false",
            "--rotate_image=false",
            "--invert=false",
            "--white_noise=false",
            "--smooth_noise=false",
            "--blur=false",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except Exception:
            return False

        return os.path.exists("_test_font.tif")
    finally:
        os.chdir(original_dir)


# проверка шрифтов
print("Проверка шрифтов...")
working_fonts = []
for font in FONTS:
    if font_works(font):
        working_fonts.append(font)
        print(f"  OK: {font}")

print(f"\nРабочих шрифтов: {len(working_fonts)}")
if not working_fonts:
    raise SystemExit("ERROR: ни один шрифт не работает!")

#слова
WORDS = [
    "Kinder", "Chocolate", "Coffee", "Store", "Michael", "Kors",
    "Original", "Product", "Quality", "Best", "Price", "Milk",
    "Sugar", "Water", "Juice", "Bread", "Cheese", "Apple",
    "Orange", "Banana", "Lemon", "Lime", "Mango", "Grape",
    "Open", "Closed", "Sale", "Discount", "Free", "New",
    "Offer", "Special", "Bonus", "Gift", "Pack", "Fresh",
    "Restaurant", "Cafe", "Bar", "Hotel", "Shop", "Market",
    "Phone", "Laptop", "Tablet", "Camera", "Music", "Video",
    "Computer", "Screen", "Keyboard", "Mouse", "Speaker",
    "Moscow", "London", "Paris", "Berlin", "Rome", "Madrid",
    "Russian", "English", "German", "French", "Spanish", "Italian",
    "Welcome", "Hello", "Goodbye", "Thanks", "Please", "Sorry",
    "Yes", "No", "Maybe", "Always", "Never", "Together",
]


def random_text():
    if random.random() < 0.8:
        return random.choice(WORDS)
    return f"{random.choice(WORDS)} {random.choice(WORDS)}"


# генерация
print(f"\nГенерация {NUM_SAMPLES} образцов...")
success = 0
fail = 0

original_dir = os.getcwd()
os.chdir(TMP_DIR)

try:
    for i in range(NUM_SAMPLES):
        text = random_text()
        font = random.choice(working_fonts)
        font_size = random.randint(24, 60)

        with open("gen_input.txt", "w", encoding="utf-8") as f:
            f.write(text)

        tmp_prefix = f"gen_{i}"

        cmd = [
            "text2image",
            "--text=gen_input.txt",
            f"--outputbase={tmp_prefix}",
            f"--font={font}",
            f"--ptsize={font_size}",
            "--margin=10",
            "--resolution=72",
            "--degrade_image=false",
            "--rotate_image=false",
            "--invert=false",
            "--white_noise=false",
            "--smooth_noise=false",
            "--blur=false",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            fail += 1
            for f in glob.glob(tmp_prefix + "*"):
                try:
                    os.remove(f)
                except:
                    pass
            continue
        except Exception:
            fail += 1
            continue

        tif_file = tmp_prefix + ".tif"
        if not os.path.exists(tif_file):
            fail += 1
            for f in glob.glob(tmp_prefix + "*"):
                try:
                    os.remove(f)
                except:
                    pass
            continue

        img = cv2.imread(tif_file, cv2.IMREAD_GRAYSCALE)

        for f in glob.glob(tmp_prefix + "*"):
            try:
                os.remove(f)
            except:
                pass

        if img is None:
            fail += 1
            continue

        # Инверсия
        if np.mean(img) < 127:
            img = cv2.bitwise_not(img)

        # Бинаризация
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Обрезка
        coords = cv2.findNonZero(255 - img_bin)
        if coords is None:
            fail += 1
            continue
        x, y, w, h = cv2.boundingRect(coords)
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img_bin.shape[1] - x, w + 2 * pad)
        h = min(img_bin.shape[0] - y, h + 2 * pad)
        cropped = img_bin[y:y + h, x:x + w]
        cropped = cv2.resize(cropped, (64, 64))

        # Сохраняем (абсолютные пути!)
        cv2.imwrite(os.path.join(TRAIN_NORMAL, f"img_{i:05d}.png"), cropped)
        cv2.imwrite(os.path.join(TRAIN_ROTATED, f"img_{i:05d}.png"),
                    cv2.rotate(cropped, cv2.ROTATE_180))

        success += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{NUM_SAMPLES} — успешно: {success}, ошибок: {fail}")
finally:
    os.chdir(original_dir)

print(f"\nУспешно: {success}, Ошибок: {fail}")

# разметка
normal_count = len(glob.glob(os.path.join(TRAIN_NORMAL, "*")))
rotated_count = len(glob.glob(os.path.join(TRAIN_ROTATED, "*")))

with open(os.path.join(OUTPUT_DIR, "train.txt"), "w") as f:
    for p in sorted(glob.glob(os.path.join(TRAIN_NORMAL, "*"))):
        f.write(f"{p}\t0\n")
    for p in sorted(glob.glob(os.path.join(TRAIN_ROTATED, "*"))):
        f.write(f"{p}\t1\n")

print(f"\nИтого: {normal_count} нормальных + {rotated_count} перевёрнутых")
print("Датасет готов: dataset/")
