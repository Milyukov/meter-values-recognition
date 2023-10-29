# Counter Recognition

## Описание
Этот репозиторий реализует следующий функционал:

1. Распознавание показаний аналоговых счетчиков по фотографии.
Распознавание просиходит в два этапа: определение области экрана счетчика с учетом ее ориентации в пространстве (угловые точки, начиная с левой верхней относительно экрана, в порядке по часовой стрелке) и определение позиции и значений цифр и области цифр после запятой (стандартная задача детектирования).

2. Обучение моделей для первой и второй стадии распознавания на заранее подготовленных данных в формате tensorflow datasets.

3. Подбор гипер-параметров для обучения обеих моделей.

4. Тестирование отдельно первой стадии и совокупно обеих стадий на размеченных данных.

5. Подготовка моделей к inference.

6. Запуск web-сервера с алгоритмом распознавания, запускающимся по запросу от клиента.

7. Создание Docker-контейнера с web-сервером из п. 5 внутри.

## Установка

Для настройки среды я скачал Anaconda [отсюда](https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh).

После установки, нужно сконфигурировать Anaconda:
```cmd
conda config --append channels conda-forge
```

Создание виртуальной среды (запустить в терминале в корне репозитория):
```cmd
conda create --name counters python==3.11.5
```

Активация среды:
```cmd
conda activate counters
```

Установка библиотек:
```cmd
pip install -r requirements.txt
```

## Использование
### 1. Обучение

Чтобы запустить обучение первой стадии, нужно выполнить в терминале в корне репозитория:
```cmd
python train_stage1.py --dataset_path <path> --resume_training <flag>
```
Аргумент dataset_path указывает на путь к папке с датасетом в формате tensorflow dataset, а аргумент resume_training может принимать значения True или False в зависимости от того, нужно ли использовать предыдущие веса и параметрны обучения для инициализации модели или нужно обучить модель с нуля.

Например:

```cmd
python train_stage1.py --dataset_path /mnt/images/counters-datasets/meter_values_dataset_stage1 --resume_training True
```

Состояние модели, включая веса и параметры оптимизатора, будет сохранено в файл retinanet/stage1.keras

Чтобы запустить обучение второй стадии, нужно выполнить в терминале в корне репозитория:
```cmd
python train_stage2.py --dataset_path <path> --resume_training <flag>
```
Аргумент dataset_path указывает на путь к папке с датасетом в формате tensorflow dataset, а аргумент resume_training может принимать значения True или False в зависимости от того, нужно ли использовать предыдущие веса и параметрны обучения для инициализации модели или нужно обучить модель с нуля.

Например:

```cmd
python train_stage1.py --dataset_path /mnt/images/counters-datasets/meter_values_dataset_stage2 --resume_training True
```

Состояние модели, включая веса и параметры оптимизатора, будет сохранено в файл retinanet/stage2.keras

### 2. Подбор гипер-параметров

Прежде чем запускать обучение модели, следует подобрать гипер-параметры обучения (начальный в нашем случае шаг обучения модели на данных).

Нужно запустить скрипт, который прогонит обучение модели с разными значениями шага обучения на 5 эпохах на обучающем датасете.

Для первой стадии:
```cmd
python tune_hyper_parameters_stage1.py --dataset_path <path>
```

Для второй стадии:
```cmd
python tune_hyper_parameters_stage2.py --dataset_path <path>
```

Для того, чтобы выбрать оптимальный шаг обучения, следует запустить tensorboard в корне репозитория для визуализации исходов экспериментов.

Для первой стадии:
```cmd
tensorboard --logdir logs/stage1/hparam_tuning/ --port 6006
```

Для второй стадии:
```cmd
tensorboard --logdir logs/stage2/hparam_tuning/ --port 6006
```

Если эксперименты проходят на сервере, можно пробросить себе соотвествующий порт с помощью ssh:

```cmd
 ssh -N -f -L localhost:16006:localhost:6006 user@ip
```

В панели tensorboard следует выбрать leraning rate с лучшей динамикой обучения и ошибкой валидации:

![example](/resources/tensorboard_hp.jpg "Example of Tensorboard panel for hyper-parameters tuning")

В показанном примере лучшую ошибку дает шаг обучения 0.0001. 

Динамику можно посмотреть на графике:

![example](/resources/tensorboard_hp_plot.jpg "Example of Tensorboard panel for hyper-parameters tuning")

В показанном примере ошибка на валидации постепенно уменьшается, чего мы и хотим.

### 3. Тестирование моделей.

После обучения моделей, чтобы иметь объективную оценку качества, модели нужно протестировать.

Для тестирования модели первой стадии, нужно запустить в терминале в корне репозитория:
```cmd
python test_stage1.py --model <path/to/model.keras> --dataset <path/to/dataset> --binary <flag>
```
--model - путь до файла модели в формате keras   
--dataset - путь до данных в формате tensorflow dataset   
--binary - булевый флаг, сигнализирующий, нужны ли бинарные метрики (аналоговый или цифровой) или по классам. По умолчанию True.   

Метрики считаются как элементы confusion matrix.

Для тестирования алгритма распознавания показаний счетчиков целиком нужно запустить в терминале в корне репозитория:
```cmd
python test_both_stages.py --images <path/to/images> --labels_stage1 <path/to/stage1/labels.json> --labels_stage2 <path/to/stage2/labels.json> --dataset <path/to/stage2/dataset>
```
--images - путь до изображений в формате *.jpg    
--labels_stage1 - путь до файла labels.json, экспортированного из Label Studio и содержащего разметку изображений для первой стадии    
--labels_stage2 - путь до файла labels.json, экспортированного из Label Studio и содержащего разметку изображений для второй стадии    
--dataset - путь до данных в формате tensorflow dataset для второй стадии   

Например:
```cmd
python test_both_stages.py --images /mnt/images/counters-datasets/meter_values_dataset_stage1/stage1 --labels_stage1 /mnt/images/counters-datasets/meter_values_dataset_stage1/labels.json --labels_stage2 /mnt/images/counters-datasets/meter_values_dataset_stage2/labels.json --dataset /mnt/images/counters-datasets/meter_values_dataset_stage2
```

Скрипт выводит recognition rate, который считается как количество полностью распознанных изображений, деленное на количество всех размеченных изображений класса analog.

### 4. Подготовка моделей к inference

Обученные файлы моделей нужно положить в папку deploy/retinanet, чтобы ее содержимое выглядело следующим образом:

- stage1.keras
- stage2.keras


Для того, чтобы подготовить модели к inference, необходимо запустить в терминале в папке deploy:
```cmd
python prepare_model.py
```

В качестве результата работы скрипта будет создана папка deploy/models с моделями, готовыми к inference.


### 5. Запуск сервера

После шага 4 можно запустить web-сервер, выполнив в терминале в папке deploy:
```cmd
python serve.py
```

Сервер будет принимать POST запросы для endpoint "http://localhost:8080/recognize".   
 
<details>
 <summary><code>POST</code> <code><b>/</b></code> <code>(отправляет изображение на сервер)</code></summary>

##### Параметры

> | name      |  type     | data type               | description                                                           |
> |-----------|-----------|-------------------------|-----------------------------------------------------------------------|
> | None      |  required | image in RGB (JSON)   | N/A  |


##### Ответы

> | http code     | content-type                      | response                                                            |
> |---------------|-----------------------------------|---------------------------------------------------------------------|
> | `200`         | `application/json`        | `{"original_image_shape": list, "success": bool, "text": str, "counter_class": str, "counter_score": float, "time": float}`` 


</details>

### 6. Сборка и запуск Docker-контейнера

Чтобы собрать котейнер с web-сервером, необходимо запустить в терминале в папке deploy:
```cmd
docker build --network=host -t counter_recognition_rest_api  -f Dockerfile .
```

Чтобы запустить контейнер с web-сервером, слушающим порт 8080, необходимо запустить в терминале в папке deploy:
```cmd
docker run --rm --gpus all -p 8000:8080 counter_recognition_rest_api
```

## Поддержка

Email разработчика: cvresearch.milyukov@gmail.com

## Следующие шаги
1. Расширение функционала на цифровые счетчики (потребует отдельной логики для пост-обработки).
2. Добавление в датасет первой стадии большего количества данных типа illegible (неразличимые показания) для балансировки.
3. Ускорение модели с помощью tensorRT
4. Ускорение модели с помощью квантизации.
5. Выбор более легковесной модели для обеих стадий. Использование моделей из tensorflow model zoo.
