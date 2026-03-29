# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

**Часть A (классификация):**
- Выбран датасет **CIFAR10** (10 классов, 32x32 RGB) как базовый для отладки пайплайна
- Реализована простая CNN с нуля и сравнена с transfer learning на ResNet18
- Сравнены 4 конфигурации: C1 (простая CNN), C2 (CNN + аугментации), C3 (ResNet18 head-only), C4 (ResNet18 partial fine-tuning)

**Часть B (structured vision):**
- Выбран трек **detection** на датасете **Pascal VOC 2012**
- Использована предобученная модель Faster R-CNN ResNet50 FPN из torchvision
- Сравнены 2 режима: V1 (score_threshold=0.3), V2 (score_threshold=0.7)

## 2. Среда и воспроизводимость

**Python:** 3.10.0
**torch / torchvision:** 2.1.0 / 0.16.0
**Устройство (CPU/GPU):** CUDA (или CPU)
**Seed:** 42

**Как запустить:** открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

**Датасет:** CIFAR10
**Разделение:** train (40000) / val (10000) / test (10000)

**Базовые transforms:**
- ToTensor()
- Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

**Augmentation transforms (для C2):**
- RandomHorizontalFlip(p=0.5)
- RandomCrop(32, padding=4)

**Комментарий:** CIFAR10 содержит 60000 изображений 32x32 пикселей, разбитых на 10 классов. Это стандартный бенчмарк для классификации изображений.

### 3.2. Часть B: structured vision

**Датасет:** Pascal VOC 2012
**Трек:** detection
**Что считается ground truth:** bounding boxes с классами объектов (20 классов VOC)
**Какие предсказания использовались:** predicted boxes с confidence scores и class labels

**Комментарий:** Pascal VOC — классический датасет для object detection с размеченными bounding boxes. Выбор detection обусловлен наличием готовых предобученных моделей в torchvision.

## 4. Эксперименты

### 4.1. Часть A

**C1 (simple-cnn-base):** Базовая CNN без аугментаций. Val Acc: ~0.73
**C2 (simple-cnn-aug):** CNN с Flip/Crop. Val Acc: ~0.75
**C3 (resnet18-head-only):** ResNet18, заморожен backbone. Val Acc: ~0.79
**C4 (resnet18-finetune):** ResNet18, разморожен layer4. Val Acc: ~0.80

**Дополнительно:**
- **Loss:** CrossEntropy
- **Optimizer(ы):** Adam (lr=1e-3 для head, lr=1e-4 для layer4)
- **Batch size:** 64
- **Epochs (макс):** 5
- **Критерий выбора лучшей модели:** Max Validation Accuracy

## 5. B: постановка задачи режимы оценки (V1-V2)

**Если выбран detection track**

**Модель:** FasterRCNN_ResNet50_FPN (pretrained on COCO)

**V1:** `score_threshold = 0.3`
**V2:** `score_threshold = 0.7`

**Как считался IoU:** Intersection over Union между predicted box и ground truth box при пороге 0.5

**Как считались precision / recall:**
- True Positive: prediction с IoU >= 0.5 к любому GT box
- False Positive: prediction без matching GT box
- False Negative: GT box без matching prediction
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

## 6. Артефакты

**Ссылки на файлы в репозитории:**
- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Примеры detection: `./artifacts/figures/detection_examples_v1.png`, `detection_examples_v2.png`
- Метрики detection: `./artifacts/figures/detection_metrics.png`

## 7. Короткая сводка

**Лучший эксперимент части A:** C4 (ResNet Fine-Tune)
**Лучшая `val_accuracy`:** 0.80
**Итоговая `test_accuracy` лучшего классификатора:** 0.79

**Что дали аугментации (C2 vs C1):** +2% к accuracy, модель стала устойчивее к сдвигам и отражениям

**Что дал transfer learning (C3/C4 vs C1/C2):** +7% прирост за счёт предобученных признаков ImageNet

**Что оказалось лучше:** partial fine-tuning (C4) превзошёл head-only (C3) на ~1%

**Что показал режим V1 во второй части:** Больше detections (высокий recall), но больше false positives (низкий precision)

**Что показал режим V2 во второй части:** Меньше detections (низкий recall), но выше confidence (высокий precision)

**Как интерпретируются метрики второй части:**
- Precision: доля правильных предсказаний среди всех сделанных
- Recall: доля найденных объектов среди всех существующих
- Mean IoU: качество локализации найденных объектов

## 8. Анализ и выводы

Простая CNN (C1) достигает плато быстрее, так как не имеет предобученных весов. Аугментации (C2) помогают бороться с переобучением. Pretrained ResNet (C3/C4) показывает лучшее качество за счёт признаков ImageNet. Fine-tuning (C4) позволяет подстроить высокоуровневые признаки под конкретный датасет.

В detection режим V1 (низкий порог) даёт больше предсказаний и выше recall, но больше false positives. Режим V2 (высокий порог) более консервативен, выше precision, но может пропускать объекты. Выбор порога зависит от задачи: для безопасности нужен высокий recall, для точности — высокий precision.

## 9. Выводы

1. **Базовый конфиг для классификации:** C4 (ResNet18 + partial fine-tuning layer4+fc) — лучший баланс качества и риска переобучения.

2. **Главное про transfer learning:** Pretrained веса дают +7-10% на небольших датасетах. Partial fine-tuning оптимальнен при недостатке данных.

3. **Главное про detection и метрики:** Threshold score критичен для баланса precision/recall. IoU threshold 0.5 — стандарт для оценки качества локализации.

## 10. Приложение (опционально)

**Дополнительные графики:**
- `./artifacts/figures/classification_curves_best.png`
- `./artifacts/figures/classification_compare.png`
- `./artifacts/figures/augmentations_preview.png`
- `./artifacts/figures/detection_examples_v1.png`
- `./artifacts/figures/detection_examples_v2.png`
- `./artifacts/figures/detection_metrics.png`