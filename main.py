import cv2
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from icecream import icecream


def extract_page_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError("Filename does not contain a page number")


def detect_page_boundaries(image, page_number):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center_col = width // 2

    def find_boundary(array):
        for i in range(len(array) - 10):
            if array[i] > 120 and array[i + 10] > 120:
                return i
        return -1

    top_boundary = find_boundary(gray[:, center_col])
    bottom_boundary = height - find_boundary(gray[::-1, center_col])
    boundary_offset = int(2 * 118.11)  # Примерно 236 пикселей

    if page_number % 2 == 0:
        left_boundary = find_boundary(gray[height // 2, :])
        right_boundary = width - boundary_offset
    else:
        right_boundary = width - find_boundary(gray[height // 2, ::-1])
        left_boundary = boundary_offset

    if top_boundary < 50 or bottom_boundary > (height - 50) or \
            left_boundary < 50 or right_boundary > (width - 50):
        return None

    return top_boundary, bottom_boundary, left_boundary, right_boundary


def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        page_number = extract_page_number(os.path.basename(image_path))
        boundaries = detect_page_boundaries(image, page_number)
        if boundaries is None:
            return None, image_path
        return boundaries, image_path
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, image_path


def find_average_frame_parallel(image_paths, source_folder_name, threshold=400):
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_image, image_paths))

    valid_results = [r for r, path in results if r is not None]
    if not valid_results:
        print("No valid results found.")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for _, path in results:
                futures.append(executor.submit(save_unprocessed, path, unprocessed_folder, source_folder_name))
            for future in futures:
                future.result()
        return None, []

    first_page_boundaries = None
    for r, path in results:
        if r is not None:
            first_page_boundaries = r
            first_path = path
            break

    if first_page_boundaries is None:
        print("No valid results found after filtering.")
        return None, [path for _, path in results]

    first_page_height = first_page_boundaries[1] - first_page_boundaries[0]
    first_page_width = first_page_boundaries[3] - first_page_boundaries[2]

    final_results = [(r, path) for r, path in results if r is not None and
                     abs((r[1] - r[0]) - first_page_height) < threshold and
                     abs((r[3] - r[2]) - first_page_width) < threshold]
    outliers = [path for _, path in results if _ is None or (
            abs((_[1] - _[0]) - first_page_height) >= threshold or
            abs((_[3] - _[2]) - first_page_width) >= threshold)]

    if not final_results:
        print("No final results found after filtering.")
        return None, outliers

    avg_top = int(np.mean([r[0] for r, _ in final_results]))
    avg_bottom = int(np.mean([r[1] for r, _ in final_results]))
    avg_left = int(np.mean([r[2] for r, _ in final_results]))
    avg_right = int(np.mean([r[3] for r, _ in final_results]))

    elapsed_time = time.time() - start_time
    print(f"find_average_frame_parallel took {elapsed_time:.2f} seconds")

    return (avg_top, avg_bottom, avg_left, avg_right), outliers


def crop_and_save(image_path, frame, output_folder_tiff, output_folder_jpg, source_folder_name):
    start_time = time.time()
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    top, bottom, left, right = frame

    page_number = extract_page_number(os.path.basename(image_path))

    # Центрируем рамку
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    frame_n_top = center_y - (bottom - top) // 2
    frame_n_bottom = center_y + (bottom - top) // 2
    frame_n_left = center_x - (right - left) // 2
    frame_n_right = center_x + (right - left) // 2

    # Увеличиваем рамку для TIFF
    cm_to_pixels = int(0.8 * 118.11)
    if page_number % 2 == 0:
        tiff_left = max(0, frame_n_left - cm_to_pixels)
        tiff_right = min(width, frame_n_right + cm_to_pixels * 2)
    else:
        tiff_left = max(0, frame_n_left - cm_to_pixels * 2)
        tiff_right = min(width, frame_n_right + cm_to_pixels)
    tiff_top = max(0, frame_n_top - cm_to_pixels)
    tiff_bottom = min(height, frame_n_bottom + cm_to_pixels)
    cropped_tiff = image[tiff_top:tiff_bottom, tiff_left:tiff_right]

    if cropped_tiff.size == 0:
        print(f"Error: Cropped TIFF image is empty for {image_path}")
        return

    cm_to_pixels = int(0.6 * 118.11)
    cm_to_pixels_more = int(0.7 * 118.11)
    cm_to_pixels_less = int(-0.3 * 118.11)
    if page_number % 2 == 0:
        jpg_left = min(width, frame_n_left + cm_to_pixels_more)
        jpg_right = frame_n_right - cm_to_pixels_less
    else:
        jpg_left = frame_n_left + cm_to_pixels_less
        jpg_right = max(0, frame_n_right - cm_to_pixels_more)

    jpg_top = max(0, frame_n_top + cm_to_pixels)
    jpg_bottom = min(height, frame_n_bottom - cm_to_pixels)
    cropped_jpg = image[jpg_top:jpg_bottom, jpg_left:jpg_right]

    if cropped_jpg.size == 0:
        print(f"Error: Cropped JPG image is empty for {image_path}")
        return

    filename = os.path.splitext(os.path.basename(image_path))[0]
    filename_with_prefix = f"{source_folder_name} - {filename}"

    tiff_output_path = os.path.join(output_folder_tiff, filename_with_prefix + ".tif")
    jpg_output_path = os.path.join(output_folder_jpg, filename_with_prefix + ".jpg")

    cv2.imwrite(tiff_output_path, cropped_tiff)
    cv2.imwrite(jpg_output_path, cropped_jpg)
    elapsed_time = time.time() - start_time
    print(f"Saved {filename}.tif and {filename}.jpg in {elapsed_time:.2f} seconds")


def process_folder(folder_path, output_folder_tiff, output_folder_jpg, unprocessed_folder):
    start_time = time.time()
    source_folder_name = os.path.basename(folder_path)
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                   filename.endswith(".tif")]
    avg_frame, outliers = find_average_frame_parallel(image_paths, source_folder_name)
    icecream.ic(outliers)
    os.makedirs(output_folder_tiff, exist_ok=True)
    os.makedirs(output_folder_jpg, exist_ok=True)
    os.makedirs(unprocessed_folder, exist_ok=True)

    if avg_frame is None:
        print("No valid average frame could be determined. Exiting.")
        return

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for image_path in image_paths:
            boundaries, _ = process_image(image_path)
            if boundaries is None:
                futures.append(executor.submit(save_unprocessed, image_path, unprocessed_folder, source_folder_name))
            else:
                futures.append(
                    executor.submit(crop_and_save, image_path, avg_frame, output_folder_tiff, output_folder_jpg,
                                    source_folder_name))
        for future in futures:
            future.result()

    while outliers:
        print(f"Processing outliers...")
        avg_frame_outliers, new_outliners = find_average_frame_parallel(outliers, source_folder_name)
        icecream.ic(new_outliners)
        if avg_frame_outliers is None:
            print("No valid average frame for outliers. Saving all outliers as unprocessed.")
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                for image_path in outliers:
                    futures.append(executor.submit(save_unprocessed, image_path, unprocessed_folder,
                                                   source_folder_name))
                for future in futures:
                    future.result()
            break

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for image_path in outliers:
                boundaries, _ = process_image(image_path)
                if boundaries is None:
                    futures.append(executor.submit(save_unprocessed, image_path, unprocessed_folder,
                                                   source_folder_name))
                else:
                    futures.append(executor.submit(crop_and_save, image_path, avg_frame_outliers, output_folder_tiff,
                                                   output_folder_jpg,
                                                   source_folder_name))
            for future in futures:
                future.result()
            outliers = new_outliners

    total_elapsed_time = time.time() - start_time
    print(f"process_folder took {total_elapsed_time:.2f} seconds")


def save_unprocessed(image_path, unprocessed_folder, source_folder_name):
    filename = os.path.basename(image_path)
    filename_with_prefix = f"{source_folder_name} - {filename}"  # Префикс с исходной папкой
    output_path = os.path.join(unprocessed_folder, filename_with_prefix)
    image = cv2.imread(image_path)
    cv2.imwrite(output_path, image)
    print(f"Saved unprocessed image {filename_with_prefix} to {unprocessed_folder}")


if __name__ == '__main__':
    folder_path = '/Users/mac/Yandex.Disk-adnemanov@stud.kpfu.ru.localized/Загрузки/15.05.2024/208743'
    output_folder_tiff = 'result/tiff'
    output_folder_jpg = 'result/jpg'
    unprocessed_folder = 'result/unprocessed'
    process_folder(folder_path, output_folder_tiff, output_folder_jpg, unprocessed_folder)
