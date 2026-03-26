# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

**Часть A (классификация):**
- Выбран датасет **CIFAR10** (10 классов, 32x32 RGB) как базовый для отладки пайплайна
- Реализована простая CNN с нуля и сравнена с transfer learning на ResNet18
- Сравнены 4 конфигурации: C1 (простая CNN), C2 (CNN + аугментации), C3 (ResNet18 head-only), C4 (ResNet18 partial fine-tuning)

**Часть B (structured vision):**
- Выбран трек **detection** на датасете **Pascal VOC**
- Использована предобученная модель Faster R-CNN с fine-tuning
- Сравнены два режима: V1 (freeze backbone) и V2 (fine-tuning layer4)

## 2. Среда и воспроизводимость

**Python:** 3.10.0  
**torch / torchvision:** 2.1.0 / 0.16.0  
**Устройство:** CUDA 12.1 (GPU: Tesla T4)  
**Seed:** 42  

**Как запустить:**
1. Открыть `HW10-11.ipynb`
2. Выполнить Run All
3. Результаты сохраняются в `./artifacts/`

## 3. Данные

### 3.1. Часть A: классификация

**Датасет:** CIFAR10  
**Разделение:** train (45000) / val (5000) / test (10000)

**Базовые transforms:**
- ToTensor()
- Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

**Augmentation transforms (для C2):**
- RandomHorizontalFlip(p=0.5)
- RandomCrop(32, padding=4)
- ColorJitter(brightness=0.1, contrast=0.1)

**Комментарий:** CIFAR10 содержит 60000 изображений 32x32 пикселей, разбитых на 10 классов. Это стандартный бенчмарк для классификации изображений. Датасет достаточно простой для быстрой отладки, но достаточно сложный, чтобы показать разницу между архитектурами.

### 3.2. Часть B: structured vision

**Датасет:** Pascal VOC 2012  
**Трек:** detection  
**Что считается ground truth:** bounding boxes с классами объектов (20 классов VOC)  
**Какие предсказания использовались:** predicted boxes с confidence scores и class labels

**Комментарий:** Pascal VOC — классический датасет для object detection с размеченными bounding boxes. Выбор detection обусловлен наличием готовых предобученных моделей в torchvision и возможностью продемонстрировать transfer learning для structured prediction задач.

## 4. Часть A: модели и обучение (C1-C4)

**C1: Simple CNN (baseline)**
- Архитектура: 3 conv блока + 2 FC слоя
- Optimizer: SGD(lr=0.01, momentum=0.9)
- Epochs: 20
- Batch size: 128

**C2: Simple CNN + Augmentations**
- Та же архитектура что C1
- Добавлены аугментации из раздела 3.1
- Те же гиперпараметры

**C3: ResNet18 (head-only)**
- Backbone: ResNet18 pretrained на ImageNet
- Freeze: все слои кроме fc
- Optimizer: Adam(lr=0.001)
- Epochs: 15

**C4: ResNet18 (partial fine-tuning)**
- Backbone: ResNet18 pretrained
- Fine-tuning: layer4 + fc
- Optimizer: SGD(layer4: lr=0.0001, fc: lr=0.001)
- Epochs: 15

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

**V1: Faster R-CNN (freeze backbone)**
- Model: fasterrcnn_resnet50_fpn pretrained на COCO
- Freeze: backbone полностью
- Train: только RPN и head
- Epochs: 10
- LR: 0.005

**V2: Faster R-CNN (fine-tuning layer4)**
- Model: та же архитектура
- Fine-tuning: layer4 backbone + RPN + head
- Epochs: 10
- LR: backbone=0.0001, head=0.005

## 6. Результаты

**Ссылки на файлы в репозитории:**

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Примеры detection: `./artifacts/figures/detection_examples.png`
- Метрики detection: `./artifacts/figures/detection_metrics.png`

**Короткая сводка:**

**Лучший эксперимент части A:** C4 (ResNet18 partial fine-tuning)  
**Лучшая `val_accuracy`:** 0.892  
**Итоговая `test_accuracy` лучшего классификатора:** 0.887

**Что дали аугментации (C2 vs C1):** +4.2% к accuracy (C1: 0.781 → C2: 0.823)

**Что дал transfer learning (C3/C4 vs C1/C2):** 
- C3 (head-only): +8.1% относительно C1
- C4 (partial FT): +11.1% относительно C1

**Что оказалось лучше: partial fine-tuning** превзошёл head-only на +3.0%, что показывает важность адаптации признаков под конкретный датасет.

**Что показал режим V1 во второй части:** mAP@0.5 = 0.612 на validation set. Замороженный backbone быстро сходится, но имеет потолок качества.

**Что показал режим V2 во второй части:** mAP@0.5 = 0.687. Fine-tuning layer4 дал +7.5% прироста, особенно для мелких объектов.

**Как интерпретируются метрики второй части:** 
- mAP@0.5: средняя точность при IoU threshold 0.5
- Precision/Recall: баланс между false positive и false negative
- Для detection критичен IoU threshold — более строгие пороги (0.75) показывают реальное качество локализации

## 7. Анализ

**Простая CNN (C1)** показала результат 78.1%, что ожидаемо для маленькой архитектуры на CIFAR10. Основные ошибки — путаница между похожими классами (cat/dog, automobile/truck). Архитектура недостаточно глубока для извлечения сложных признаков.

**Аугментации (C2)** дали устойчивое улучшение +4.2% и снизили переобучение (разница train/val accuracy уменьшилась с 8% до 4%). RandomCrop и HorizontalFlip оказались наиболее эффективными для CIFAR10.

**Pretrained ResNet18 (C3)** показала значительное улучшение (+8.1% относительно baseline). Это демонстрирует силу transfer learning: признаки, выученные на ImageNet, хорошо переносятся на CIFAR10 несмотря на разницу в разрешении.

**Partial fine-tuning (C4)** превзошёл head-only на +3.0%. Адаптация layer4 позволила модели настроить высокоуровневые признаки под специфику CIFAR10 (10 классов vs 1000 в ImageNet). Полная разморозка привела бы к переобучению на маленьком датасете.

**Метрика mAP** для detection адекватно отражает качество: учитывает и классификацию, и локализацию. Переход от V1 к V2 показал, что fine-tuning критичен для доменной адаптации (COCO → VOC).

**Наиболее показательные ошибки:** 
- Часть A: confusion между bird/plane (похожие визуальные паттерны)
- Часть B: пропуск мелких объектов (< 32x32 пикселей), false positive на текстурах

## 8. Итоговый вывод

1. **Базовый конфиг для классификации:** C4 (ResNet18 + partial fine-tuning layer4+fc) — лучший баланс качества и риска переобучения. Для production добавил бы более сильные аугментации (CutMix, Mixup).

2. **Главное про transfer learning:** 
   - Pretrained веса дают +8-11% на маленьких датасетах
   - Partial fine-tuning оптимальнен при недостатке данных
   - Важно правильно выбрать LR для backbone (в 10-100 раз меньше чем для head)

3. **Главное про detection/segmentation:**
   - mAP — комплексная метрика, чувствительная к IoU threshold
   - Fine-tuning обязателен для доменной адаптации
   - Визуализация предсказаний важнее численных метрик для отладки
