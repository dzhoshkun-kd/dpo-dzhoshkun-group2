# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- **Часть A:** Выбран датасет **CIFAR100** — содержит 100 классов объектов, изображения 32x32 RGB. Это стандартный бенчмарк для классификации, достаточно сложный для демонстрации преимуществ transfer learning.
- **Часть B:** Выбран датасет **Pascal VOC 2007**, трек **detection**. Содержит размеченные bounding boxes для 20 классов объектов. Detection выбран как более универсальная задача с готовыми pretrained-моделями в torchvision.
- **Сравнение:** В части A сравнивались 4 конфигурации (C1-C4): простая CNN без аугментаций, CNN с аугментациями, ResNet18 с замороженным backbone, ResNet18 с fine-tuning. В части B сравнивались 2 режима инференса с разными порогами уверенности (V1: 0.3, V2: 0.7).

## 2. Среда и воспроизводимость

- **Python:** 3.10.0
- **torch / torchvision:** 2.1.0 / 0.16.0
- **Устройство (CPU/GPU):** CUDA (или CPU)
- **Seed:** 42
- **Как запустить:** открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- **Датасет:** CIFAR100
- **Разделение:** train (40000) / val (10000) / test (10000) — официальный test set, train/val разделены 80/20 с фиксированным seed
- **Базовые transforms:** ToTensor(), Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
- **Augmentation transforms:** RandomHorizontalFlip(p=0.5), RandomCrop(32, padding=4)
- **Комментарий:** CIFAR100 содержит 60000 изображений 32x32 пикселей, разбитых на 100 классов. Это в 10 раз больше классов чем в CIFAR10, что делает задачу существенно сложнее и требует более мощных моделей или transfer learning.

### 3.2. Часть B: structured vision

- **Датасет:** Pascal VOC 2007
- **Трек:** detection
- **Что считается ground truth:** Bounding boxes с классами объектов (20 классов VOC) в формате XML аннотаций
- **Какие предсказания использовались:** Predicted boxes с confidence scores и class labels от Faster R-CNN
- **Комментарий:** Pascal VOC — классический бенчмарк для object detection с качественной разметкой. Выбор detection обусловлен наличием готовых pretrained-моделей в torchvision и возможностью наглядно продемонстрировать trade-off между precision и recall.

## 4. Часть A: модели и обучение (C1-C4)

- **C1 (simple-cnn-base):** Простая CNN с 3 сверточными слоями (32→64→128 фильтров), обучается с нуля без аугментаций. Val Accuracy: 0.3915
- **C2 (simple-cnn-aug):** Та же архитектура CNN, но с аугментациями (flip, crop) на обучении. Val Accuracy: 0.3733
- **C3 (resnet18-head-only):** ResNet18 с pretrained weights на ImageNet, backbone заморожен, обучается только классификационная голова (fc). Val Accuracy: 0.5774
- **C4 (resnet18-finetune):** ResNet18 с pretrained weights, разморожены layer4 + fc, разные learning rate для слоёв. Val Accuracy: 0.7084

**Дополнительно:**

- **Loss:** CrossEntropyLoss
- **Optimizer(ы):** Adam (lr=1e-3 для C1-C3 и fc в C4, lr=1e-4 для layer4 в C4)
- **Batch size:** 64
- **Epochs (макс):** 5
- **Критерий выбора лучшей модели:** Max Validation Accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран detection track

- **Модель:** FasterRCNN_ResNet50_FPN (pretrained on COCO)
- **V1:** `score_threshold = 0.3`
- **V2:** `score_threshold = 0.7`
- **Как считался IoU:** Intersection over Union между predicted box и ground truth box при пороге 0.5 для matching
- **Как считались precision / recall:** TP = prediction с IoU >= 0.5 к любому GT box, FP = prediction без matching GT, FN = GT без matching prediction. Precision = TP/(TP+FP), Recall = TP/(TP+FN)

## 6. Результаты

**Ссылки на файлы в репозитории:**

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/detection_examples.png`, `./artifacts/figures/detection_metrics.png`

**Короткая сводка:**

- **Лучший эксперимент части A:** C4 (ResNet18 + partial fine-tuning layer4+fc)
- **Лучшая `val_accuracy`:** 0.7084
- **Итоговая `test_accuracy` лучшего классификатора:** 0.7061
- **Что дали аугментации (C2 vs C1):** Не дали улучшения (0.3733 vs 0.3915) — вероятно, 5 эпох недостаточно для проявления эффекта
- **Что дал transfer learning (C3/C4 vs C1/C2):** Существенный прирост +18-31% благодаря предобученным признакам ImageNet
- **Что оказалось лучше:** Partial fine-tuning (C4) превзошёл head-only (C3) на +13.1%
- **Что показал режим V1 во второй части:** Высокий recall (0.9044), но низкий precision (0.2727) — много detections, включая false positives
- **Что показал режим V2 во второй части:** Выше precision (0.5223), чуть ниже recall (0.8603) — более консервативные предсказания
- **Как интерпретируются метрики второй части:** Precision — доля правильных предсказаний, Recall — доля найденных объектов, Mean IoU — качество локализации

## 7. Анализ

Простая CNN (C1/C2) показывает ограниченную производительность на CIFAR100 из-за отсутствия предобученных признаков и относительно маленькой архитектуры. 100 классов — сложная задача для обучения с нуля на 5 эпохах. Аугментации (C2) не дали устойчивого улучшения, что вероятно связано с недостаточным количеством эпох — аугментации требуют больше времени обучения для проявления регулярризующего эффекта.

Pretrained ResNet18 (C3) показала значительное улучшение (+18.6% к C1) благодаря признакам, обученным на ImageNet. Это подтверждает эффективность transfer learning на небольших датасетах. Partial fine-tuning (C4) превзошёл head-only (C3) на +13.1%, что показывает важность адаптации высокоуровневых признаков под конкретную задачу.

Для detection выбранные метрики (Precision/Recall/IoU) хорошо подходят для оценки качества детекции объектов. Переход от V1 к V2 демонстрирует классический trade-off: повышение порога уверенности уменьшает количество false positives (растёт precision), но может пропускать сложные объекты (снижается recall). Mean IoU остаётся высоким в обоих режимах (~80%), что говорит о хорошем качестве локализации найденных объектов.

Наиболее показательные ошибки модели: false positives на V1 — модель детектирует объекты там, где их нет (низкая уверенность); false negatives на V2 — модель пропускает частично видимые или мелкие объекты.

## 8. Итоговый вывод

- **Базовый конфиг для классификации:** C4 (ResNet18 + partial fine-tuning layer4+fc) — лучший баланс качества и риска переобучения для небольших датасетов.
- **Главное про transfer learning:** Pretrained веса дают +18-31% на CIFAR100 по сравнению с обучением с нуля. Partial fine-tuning оптимальнен при недостатке данных — позволяет подстроить высокоуровневые признаки без риска разрушить низкоуровневые.
- **Главное про detection и метрики:** Threshold score критичен для баланса precision/recall. Выбор порога зависит от задачи: для безопасности нужен высокий recall (V1), для точности — высокий precision (V2). IoU threshold 0.5 — стандарт для оценки качества локализации.
