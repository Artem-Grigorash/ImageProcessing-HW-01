import os
from datetime import datetime
from pathlib import Path
from time import time
# Import the required library
from google_images_download import google_images_download

# --- CONFIGURATION ---
TARGET = "macbook"
N_IMAGES = 200  # Вернулись к 1000
# ПУТЬ К CHROME DRIVER: Укажите полный путь к chromedriver.exe или chromedriver,
# если он установлен. Оставьте None, если он находится в вашем PATH.
CHROME_DRIVER_PATH = "/opt/homebrew/bin/chromedriver"


# Пример пути: CHROME_DRIVER_PATH = "/usr/local/bin/chromedriver"

def ensure_dir(path: Path):
    """Creates the directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def main():
    """Main function implementing the download logic."""

    # --- 1. Initialization and Directory Setup ---
    target_clean = TARGET.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Construct the final unique output path
    out_dir_path = Path(f"data/{target_clean}-{N_IMAGES}-{timestamp}")
    ensure_dir(out_dir_path)

    # The library will create a subdirectory named after the keyword inside out_dir_path
    final_save_path = out_dir_path / TARGET

    print("-" * 50)
    print(f"Target: {TARGET}")
    print(f"Saving to: {out_dir_path}")
    print(f"Images requested: {N_IMAGES}")
    print("-" * 50)

    # --- 2. Download via google_images_download ---
    response = google_images_download.googleimagesdownload()

    arguments = {
        "keywords": TARGET,
        "limit": N_IMAGES,
        # We pass the Path object converted to string as the root directory
        "output_directory": str(out_dir_path),
        # The library requires the image_directory parameter, often set to the keyword
        "image_directory": TARGET,
        "print_urls": False,
        "delay": 1,
        "silent_mode": False
    }

    # Добавление пути к драйверу, как предложило сообщение об ошибке
    if CHROME_DRIVER_PATH:
        arguments["chromedriver"] = CHROME_DRIVER_PATH

    start_time = time()

    print("Starting download using google_images_download...")

    try:
        # The library executes the scraping and prints its own progress
        paths = response.download(arguments)

        end_time = time()

        # --- 3. Final Reporting ---
        print("\n" + "-" * 30)
        print("Сканирование и скачивание завершено.")

        total_downloaded = 0
        if paths and paths[0]:
            total_downloaded = sum(len(file_list) for keyword, file_list in paths[0].items())

        print(f"Всего запрошено файлов: {N_IMAGES}")
        print(f"Всего скачано файлов: {total_downloaded}")
        print(f"Время выполнения: {end_time - start_time:.2f} секунд")
        print(f"Изображения находятся в папке: {final_save_path.resolve()}")

    except Exception as e:
        print(f"\nОшибка при скачивании: {e}")
        print("Убедитесь, что библиотека google_images_download установлена (pip install google_images_download).")


if __name__ == "__main__":
    main()
