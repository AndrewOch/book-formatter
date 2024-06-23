import cv2
import os
import re


def extract_page_number(filename):
    # Извлекаем номер страницы из названия файла
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError("Filename does not contain a page number")


def detect_page_boundaries(image_path, page_number):
    # Загрузим изображение
    image = cv2.imread(image_path)
    # Преобразуем его в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    center_col = width // 2

    # Найдем верхнюю границу
    top_boundary = 0
    for i in range(height):
        if gray[i, center_col] > 100:  # Порог для определения перехода
            top_boundary = i
            break

    # Найдем нижнюю границу
    bottom_boundary = height
    for i in range(height - 1, -1, -1):
        if gray[i, center_col] > 100:  # Порог для определения перехода
            bottom_boundary = i
            break

    # Определим левую или правую границу в зависимости от номера страницы
    boundary_offset = int(2 * 118.11)  # Примерно 236 пикселей
    if page_number % 2 == 0:
        # Найдем левую границу
        left_boundary = 0
        for i in range(width):
            if gray[height // 2, i] > 100:  # Порог для определения перехода
                left_boundary = i
                break
        right_boundary = width - boundary_offset
        return top_boundary, bottom_boundary, left_boundary, right_boundary
    else:
        # Найдем правую границу
        right_boundary = width
        for i in range(width - 1, -1, -1):
            if gray[height // 2, i] > 100:  # Порог для определения перехода
                right_boundary = i
                break
        left_boundary = boundary_offset
        return top_boundary, bottom_boundary, left_boundary, right_boundary


def find_min_frame(folder_path):
    min_top = float('inf')
    max_bottom = float('-inf')
    min_left = float('inf')
    max_right = float('-inf')

    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            image_path = os.path.join(folder_path, filename)
            try:
                page_number = extract_page_number(filename)
                top, bottom, left, right = detect_page_boundaries(image_path, page_number)

                min_top = min(min_top, top)
                max_bottom = max(max_bottom, bottom)
                min_left = min(min_left, left)
                max_right = max(max_right, right)
            except ValueError as e:
                print(f"Skipping {filename}: {e}")

    return min_top, max_bottom, min_left, max_right


def crop_and_save(image_path, min_frame, output_folder_tiff, output_folder_jpg):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    min_top, max_bottom, min_left, max_right = min_frame

    # Рамка N
    center_x = (min_left + max_right) // 2
    center_y = (min_top + max_bottom) // 2
    frame_n_top = center_y - (max_bottom - min_top) // 2
    frame_n_bottom = center_y + (max_bottom - min_top) // 2
    frame_n_left = center_x - (max_right - min_left) // 2
    frame_n_right = center_x + (max_right - min_left) // 2

    # Обрезка для tiff файла (+1 см со всех сторон)
    cm_to_pixels = int(0.5 * 118.11)
    tiff_top = max(0, frame_n_top - cm_to_pixels)
    tiff_bottom = min(height, frame_n_bottom + cm_to_pixels)
    tiff_left = max(0, frame_n_left - cm_to_pixels)
    tiff_right = min(width, frame_n_right + cm_to_pixels)
    cropped_tiff = image[tiff_top:tiff_bottom, tiff_left:tiff_right]

    # Обрезка для jpg файла (-0.5 см со всех сторон с поправками)
    page_number = extract_page_number(os.path.basename(image_path))
    cm_to_pixels = int(0.5 * 118.11)

    if page_number % 2 == 0:
        # Для четных страниц
        cm_to_pixels_more = int(0.8 * 118.11)
        cm_to_pixels_less = int(0.2 * 118.11)
        jpg_left = max(0, frame_n_left + cm_to_pixels_more)
        jpg_right = min(width, frame_n_right - cm_to_pixels_less)
    else:
        # Для нечетных страниц
        cm_to_pixels_more = int(0.8 * 118.11)
        cm_to_pixels_less = int(0.2 * 118.11)
        jpg_left = max(0, frame_n_left + cm_to_pixels_less)
        jpg_right = min(width, frame_n_right - cm_to_pixels_more)

    jpg_top = max(0, frame_n_top + cm_to_pixels)
    jpg_bottom = min(height, frame_n_bottom - cm_to_pixels)
    cropped_jpg = image[jpg_top:jpg_bottom, jpg_left:jpg_right]

    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Сохранение файлов
    tiff_output_path = os.path.join(output_folder_tiff, filename + ".tif")
    jpg_output_path = os.path.join(output_folder_jpg, filename + ".jpg")

    cv2.imwrite(tiff_output_path, cropped_tiff)
    cv2.imwrite(jpg_output_path, cropped_jpg)
    print(f"Saved {filename}.tif and {filename}.jpg")


def process_folder(folder_path, output_folder_tiff, output_folder_jpg):
    min_frame = find_min_frame(folder_path)
    os.makedirs(output_folder_tiff, exist_ok=True)
    os.makedirs(output_folder_jpg, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            image_path = os.path.join(folder_path, filename)
            try:
                crop_and_save(image_path, min_frame, output_folder_tiff, output_folder_jpg)
                print(f"Processed {filename}")
            except ValueError as e:
                print(f"Skipping {filename}: {e}")


folder_path = 'data'
output_folder_tiff = 'result/tiff'
output_folder_jpg = 'result/jpg'
process_folder(folder_path, output_folder_tiff, output_folder_jpg)
