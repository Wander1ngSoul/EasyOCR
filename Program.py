import os
import cv2
import easyocr
import re
import matplotlib.pyplot as plt

reader = easyocr.Reader(['ru'])

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def process_image(image_path):
    image = cv2.imread(image_path)
    gray_image = convert_to_grayscale(image)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    result = reader.readtext(gray_image)
    print(f"\nОбработка изображения: {image_path}")
    combined_numbers = ""
    for detection in result:
        text = detection[1]
        text = text.replace('@', '0')
        numbers = re.sub(r'[^0-9.,]', '', text)
        if numbers:
            combined_numbers += numbers
    print(f"Числа: {combined_numbers}")
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.title("Оригинал")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title("Оттенки серого")
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.title("Бинаризация")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title("Распознанный текст")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for detection in result:
        top_left = tuple(map(int, detection[0][0]))
        bottom_right = tuple(map(int, detection[0][2]))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

folder_path = 'all_data/ru_val'

for image_name in os.listdir(folder_path):
    if image_name.endswith('.jpeg'):
        image_path = os.path.join(folder_path, image_name)
        process_image(image_path)
