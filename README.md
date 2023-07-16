# Пример просто автоэнкодера и обучения классификатора на латентных представлениях данных


Создаем окружение

`conda create -n "enc_env" python=3.8.12`

`conda activate enc_env`

Клонируем репозиторий

`git clone https://github.com/dkorzh10/autoencoder.git`

Переходим в директорию

`cd autoencoder`

Устанваливаем зависимости

`pip install -r requirements.txt`

`pip intall -e .`

Открываем ноутбук `notebooks/report.ipynb` и работаем.

## Результаты
|Метрика      | Значение |
| ---      | ---       |
| Гипотеза     | Обычный сверточный энкодер,<br />Transposed Convlution декодер     |
| MSE Test loss autoencoder | `5.5e-5`         |
| Test Accuracy Classifier     | `0.5442`        |