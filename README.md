# Cat2Tiger
CycleGAN "раскрашивающая" кошек под тигров. 

Было интересно разобраться с CycleGAN, но использовать готовый датасет не интересно. Так как сейчас год тигра, было принято решение сделать сеть которая превращается изображение и видео с кошками в тигров. 

## Пример работы сети после 28 эпох.
![Real image](/samples/GIF_real.gif)
![Fake image](/samples/GIF_fake.gif)

Благодарю за предоставленное видео Anastasia Shuraeva с Pexels [pexels-anastasia-shuraeva](https://www.pexels.com/video/a-video-of-a-cat-yawning-7672693/).

## Использование сети.
Для использования сети достаточно запустить скрипт ***converter.py***. Скрипт ищет в папке файл image.jpg или video.mp4 и пропускает его через сеть. Для конвертирования видео в GIF запустите скрипт с флагом "--output gif"

## Сбор датасета.
Датасет был собран с помощью поискака изображений BING. Для сбора собственного датасета достаточно выполнить ноутбук ***GetImages.ipynb***. Для сбора датасета использукется Selenium и Chromedrive

## Обучение модели.
Обучение реальзовано в ***CycleGAN.ipynb***. Ноутбук оптимизирован для обучения на Colab. В ноутбуке реализовано обучение как с нуля так и дообучение. Для обучения с 0 эпохи установите параметр start_epoch = 0. Веса сохраняются в папке ***weights***.
