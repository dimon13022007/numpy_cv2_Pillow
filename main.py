import sqlite3
from PIL import Image
import cv2
import numpy as np


def extract_edge_pixels(image_path):
    # Загрузка изображения в оттенках серого
    image = cv2.imread(image_path, 0)
    if image is None:
        print("Ошибка загрузки изображения.")
        return

    # Применение оператора Кэнни для обнаружения границ
    edges = cv2.Canny(image, 100, 200)

    # Получение координат краевых пикселей
    edge_pixels = np.where(edges != 0)

    return edge_pixels



def extract_deviant_pixels(image_path, deviation_threshold, deviation_interval):
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка загрузки изображения.")
        return

    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Вычисление среднего значения и стандартного отклонения
    mean_value = np.mean(gray_image)
    std_value = np.std(gray_image)

    # Определение пороговых значений для отклонения
    lower_threshold = mean_value - deviation_interval * std_value
    upper_threshold = mean_value + deviation_interval * std_value

    # Выделение пикселей, значение которых отклоняется от среднего значения более заданного интервала
    deviant_pixels = np.where((gray_image < lower_threshold) | (gray_image > upper_threshold))

    return deviant_pixels



def extract_cloudy_region(image_path, binary_image):
    # Загрузка оригинального изображения
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка загрузки изображения.")
        return

    # Преобразование бинарного изображения в массив numpy
    binary_np = np.array(binary_image, dtype=np.uint8)

    # Применение маски облаков к оригинальному изображению
    cloudy_image = cv2.bitwise_and(image, image, mask=binary_np)

    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(cloudy_image, cv2.COLOR_BGR2GRAY)

    # Применение пороговой сегментации для выделения области с облаками
    _, thresh_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Поиск контуров области с облаками
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создание маски для области с облаками
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Выделение области с облаками на оригинальном изображении
    cloudy_region_image = cv2.bitwise_and(image, mask)

    # Сохранение изображения области с облаками
    cloudy_region_image_path = "cloudy_region_image.jpg"
    cv2.imwrite(cloudy_region_image_path, cloudy_region_image)

    # Определение пикселей, составляющих область с облаками
    pixels = cv2.findNonZero(mask)

    return cloudy_region_image_path, pixels



def extract_cloudless_region(image_path, binary_image):
    # Загрузка оригинального изображения
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка загрузки изображения.")
        return

    # Преобразование бинарного изображения в массив numpy
    binary_np = np.array(binary_image, dtype=np.uint8)

    # Применение маски облаков к оригинальному изображению
    cloudless_image_np = cv2.bitwise_and(image, image, mask=binary_np)

    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(cloudless_image_np, cv2.COLOR_BGR2GRAY)

    # Применение пороговой сегментации для выделения области без облаков
    _, thresh_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Поиск контуров области без облаков
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создание маски для области без облаков
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Выделение области без облаков на оригинальном изображении
    cloudless_region_image = cv2.bitwise_and(image, mask)

    # Сохранение изображения области без облаков
    cloudless_region_image_path = "cloudless_region_image.jpg"
    cv2.imwrite(cloudless_region_image_path, cloudless_region_image)

    return cloudless_region_image_path

def create_database():
    conn = sqlite3.connect("field_images.db")
    cursor = conn.cursor()

    # Создание таблицы для хранения снимков
    cursor.execute('''CREATE TABLE IF NOT EXISTS field_images
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, image_path TEXT)''')

    conn.commit()
    conn.close()

def insert_image(image_path):
    conn = sqlite3.connect("field_images.db")
    cursor = conn.cursor()

    # Загрузка изображения
    image = Image.open(image_path)
    if image is None:
        print("Ошибка загрузки изображения.")
        return

    # Преобразование изображения в оттенки серого
    gray_image = image.convert("L")

    # Применение алгоритма распознавания облаков
    # Например, можно использовать пороговую сегментацию
    threshold_value = 200
    binary_image = gray_image.point(lambda pixel: pixel > threshold_value and 255)

    # Расчет площади облаков
    cloud_pixels = binary_image.histogram()[255]
    total_pixels = binary_image.size[0] * binary_image.size[1]
    cloud_coverage = (cloud_pixels / total_pixels) * 100

    # Преобразование изображения в массив numpy
    binary_np = np.array(binary_image, dtype=np.uint8)

    # Обнаружение контуров полей
    contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создание цветного изображения для отрисовки контуров
    color_image = cv2.cvtColor(binary_np, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)

    # Сохранение изображения с контурами полей
    output_image_path = "output_image.jpg"
    cv2.imwrite(output_image_path, color_image)

    # Занесение пути к изображению в базу данных
    cursor.execute("INSERT INTO field_images (image_path) VALUES (?)", (output_image_path,))
    conn.commit()

    conn.close()

    return cloud_coverage, output_image_path

def generate_cloudless_image(image_path, binary_image):
    # Загрузка оригинального изображения
    image = Image.open(image_path)
    if image is None:
        print("Ошибка загрузки изображения.")
        return

    # Преобразование бинарного изображения в массив numpy
    binary_np = np.array(binary_image, dtype=np.uint8)

    # Применение маски облаков к оригинальному изображению
    cloudless_image_np = cv2.bitwise_and(image, image, mask=binary_np)
    cloudless_image = Image.fromarray(cloudless_image_np)

    # Сохранение изображения без облаков
    cloudless_image_path = "cloudless_image.jpg"
    cloudless_image.save(cloudless_image_path)

    return cloudless_image_path


def compare_brightness(image_path, reference_image_path):
    # Загрузка изображений
    image = cv2.imread(image_path, 0)
    reference_image = cv2.imread(reference_image_path, 0)

    if image is None or reference_image is None:
        print("Ошибка загрузки изображений.")
        return

    # Вычисление среднего значения яркости на изображениях
    mean_brightness = np.mean(image)
    reference_mean_brightness = np.mean(reference_image)

    # Вычисление разницы между средними значениями
    brightness_difference = mean_brightness - reference_mean_brightness

    # Вывод информации о затенении или покрытии облаками
    if brightness_difference > 0:
        print("Затенение земной поверхности облаками.")
    elif brightness_difference < 0:
        print("Ясная погода.")
    else:
        print("Нет затенения или покрытия облаками.")

# Пример использования
image_path = "Ваше изображение.jpg"
reference_image_path = "Эталонное изображение ясной погоды.jpg"

compare_brightness(image_path, reference_image_path)

# Пример использования
image_path = "Снимок экрана 2023-06-18 205834.png"

# Создание базы данных (выполнять только один раз)
create_database()

# Занесение снимка в базу данных и получение значений cloud_coverage и output_image_path
cloud_coverage, output_image_path = insert_image(image_path)

deviation_threshold = 30
deviation_interval = 2

# Вывод информации
print(f"Облачность: {cloud_coverage}%")
print(f"Сохраненное изображение с контурами полей: {output_image_path}")



# Выделение пикселей с отклонениями от среднего значения
deviant_pixels = extract_deviant_pixels(image_path, deviation_threshold, deviation_interval)
print("Пиксели с отклонениями от среднего значения:")
print(deviant_pixels)

# Выделение краевых пикселей
edge_pixels = extract_edge_pixels(image_path)
print("Краевые пиксели:")
print(edge_pixels)