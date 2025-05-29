# Случайные графы для проверки гипотез

![Python CI](https://github.com/berdov/dm/actions/workflows/python-ci.yml/badge.svg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f5a9534103dd4887bcc3677dacdfca1b)](https://app.codacy.com/gh/berdov/dm/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


## Описание проекта

Данный проект посвящён исследованию применения случайных графов для задач проверки гипотез о распределении данных.  
Мы используем две модели случайных графов:
— KNN-граф (по k ближайшим соседям).
- дистанционный граф (по фиксированному порогу расстояния).

Задача проекта — исследовать свойства различных характеристик этих графов и применить их для построения статистических критериев согласия.  
Исследования проводятся методом Монте-Карло.

## Структура проекта

- `src/` — исходный код.
- `notebooks/` — Jupyter ноутбуки с экспериментами.
- `report/` — отчёт в MD формате.

## Инструкция по запуску и сборке

Установите зависимости:
```bash
pip install -r requirements.txt
```

Для начала скомпилируйте файл с основными методами и функциями:
```bash
python src/graph_knn.py
```

Затем запустите ноутбуки с экспериментами или построением моделей, например:
```bash
jupyter notebook notebooks/analyze_berdov.ipynb
jupyter notebook notebooks/analyze_Вerdov.ipynb
```



## Используемые библиотеки

- numpy (BSD License)
- networkx (BSD License)
- scikit-learn (BSD License)
- matplotlib (PSF License)
- scipy (BSD License)
- pandas (BSD License)

## Лицензия

MIT License — см. файл LICENSE.
