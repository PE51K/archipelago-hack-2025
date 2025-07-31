import sys
import time
from pathlib import Path
import random
import numpy as np
import pandas as pd
from typing import List, Union


# Фиксируем сиды
SEED = 42
random.seed(SEED)
np.random.seed(SEED)



def generate_random_solution(image: np.ndarray) -> dict:
    xc = random.random() * 0.8
    yc = random.random() * 0.8
    w = random.random() * 0.2
    h = random.random() * 0.2
    conf = random.random()
    
    return {
        'xc': round(xc, 4),
        'yc': round(yc, 4),
        'w': round(w, 4),
        'h': round(h, 4),
        'label': 0,
        'score': round(conf, 4),
    }



def model_predict_one_image(image: np.ndarray) -> list:
    """
    Это пример, рандомно заполняющий результат работы модели
    
    Args:
        image (np.ndarray): изображение

    Returns:
        dict: словарь с результатами предикта на изображении
    """
    # Симуляция работы модели
    time.sleep(0.1)
    # Возвращаем пустой словарь в случае если ничего не нашли на изображении
    image_results = []
    # Эмуляция поиска 10 человек на изображении
    for _ in range(10):
        # Рандом с вероятностью 0.6 находим человека
        if random.random() > 0.6:
            image_results.append(generate_random_solution(image))
    
    return image_results


def predict(images: Union[List[np.ndarray], np.ndarray]) -> dict:
    """Функция производит инференс модели на изображении

    Args:
        image (np.ndarray): изображение, открытое через cv2 в RGB формате

    Returns:
        list[dict]: список списков словарей с результатами предикта 
        на найденных изображениях [
            [
                {
                    'xc': round(xc, 4),
                    'yc': round(yc, 4),
                    'w': round(w, 4),
                    'h': round(h, 4),
                    'label': 0,
                    'score': round(conf, 4),
                },  
                ...
            ],
                ...
            [
                {
                    'xc': round(xc, 4),
                    'yc': round(yc, 4),
                    'w': round(w, 4),
                    'h': round(h, 4),
                    'label': 0,
                    'score': round(conf, 4),
                },  
                ...
            ]
        ]
    """    
    results = []
    if isinstance(images, np.ndarray):
        images = [images]
    
    # У вас должен быть такой код:
    # results = model.predict(images)
    
    # Либо если батч = 1 изображение
    for image in images:
        # Эмуляция, что модель может ничего не найти вовсе
        if random.random() > 0.5:
            image_results = model_predict_one_image(image)
        else:
            image_results = []
        # Если модель обрабатывает максимум 1 изображение за 1 раз
        # result = model.predict(image)
        results.append(image_results)
    
    return results
