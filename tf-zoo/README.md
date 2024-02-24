## Установка

```cmd
conda create -n model_zoo python=3.11
conda activate model_zoo
pip install tf-models-official
pip install IPython
curl -L 'https://public.roboflow.com/ds/ZpYLqHeT0W?key=ZXfZLRnhsc' > './BCCD.v1-bccd.coco.zip'
unzip -q -o './BCCD.v1-bccd.coco.zip' -d './BCC.v1-bccd.coco/'
rm './BCCD.v1-bccd.coco.zip'
```

## Запуск обучения

Для запуска обучения моедли нужно выполнить следующую команду в терминале:

```cmd
python train.py --train_dataset /path/to/train.tfrecord --valid_dataset /path/to/valid.tfrecord --model_dir /path/to/output/folder
```

Чтобы посмотреть метрики в Tensorboard нужно запустить:

```cmd
tensorboard --logdir /path/to/output/folder --port 6006
```

## Подготовка модели к использованию

Для подготовки модели к использованию на стороне сервера, нужно запустить следующую команду в терминале в модуле deploy :

```cmd
python tf_zoo_deploy.py
```

предварительно положив папку с результатами обучения в модуль deploy и назвав ее `trained_model_counters`.

В результате будет сгенерирована папка `exported_model_counters`.
