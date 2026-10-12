# 📘 Практическая работа 4: Генеративно-состязательные сети (GAN)

**Генерация изображений на примере CIFAR-10**  

---

## 🎯 Цель и задачи

**Цель работы:**
Изучить принципы работы генеративно-состязательных сетей (GAN) на примере задачи генерации изображений. Сравнить производительность архитектур: базового GAN, DCGAN и опционально — более продвинутых модификаций (WGAN-GP, Conditional GAN).

**Задачи:**

1. Изучить библиотеки для работы с GAN (`TensorFlow/Keras` или `PyTorch`).
2. Загрузить и предобработать датасет CIFAR-10 (или альтернативный: CelebA, MNIST, STL-10).
3. Реализовать две базовые архитектуры:
   - **Vanilla GAN** (полносвязные слои)
   - **DCGAN** (свёрточные слои, батч-нормализация, ReLU/LeakyReLU)
4. Обучить модели, отследить динамику `loss` генератора и дискриминатора.
5. Оценить качество генерации: визуальная оценка + количественные метрики (`FID`, `Inception Score`).
6. *(Опционально)* Реализовать и сравнить с продвинутой архитектурой (WGAN-GP / Conditional GAN / StyleGAN2-ADA).
7. *(Опционально)* Сравнить с диффузионной моделью (DDPM) или VAE.
8. Визуализировать результаты: сгенерированные изображения, latent space interpolation, training dynamics.

---

## 🛠 Используемые инструменты

| Категория Рекомендуемые инструменты  |                                                      |
| ------------------------------------ | ---------------------------------------------------- |
| **Язык**                             | Python 3.9+                                          |
| **Фреймворки**                       | `TensorFlow/Keras` **или** `PyTorch`                 |
| **Обработка изображений**            | `Pillow`, `Albumentations`, `torchvision.transforms` |
| **Метрики генерации**                | `torch-fidelity`, `clean-fid`, `pytorch-fid`         |
| **Визуализация**                     | `Matplotlib`, `Seaborn`, `Plotly`, `TensorBoard`     |
| **Логирование/Оптимизация**          | `Weights & Biases`, `Optuna`, `Ray Tune`             |
| **Среда**                            | Google Colab (GPU) / Локальный Jupyter с CUDA        |

> 💡 **Рекомендация по структуре проекта:**
>
> ```
> /content/drive/MyDrive/GAN_Lab4/
> ├── data/
> │   └── cifar10/ (или celeba/)
> ├── notebooks/
> │   └── lab4_gan.ipynb
> ├── models/
> │   ├── vanilla_gan.pt
> │   ├── dcgan.pt
> │   └── wgan_gp.pt (опционально)
> ├── outputs/
>     ├── samples_epoch_XX.png
>     └── fid_scores.json
>
> ```

> ```
>
>
> ```

---

## 📋 Этапы выполнения работф

### 1. Выбор и подготовка данных

- **Датасеты:**
  - `CIFAR-10`: 60K изображений 32×32, 10 классов — базовый вариант.
  - `CelebA-HQ`: 30K лиц 1024×1024 — для продвинутых экспериментов (требует больше памяти).
  - `MNIST` / `Fashion-MNIST`: для отладки архитектуры.
- **Загрузка и нормализация:**
  ```
  # TensorFlow/Keras
  from tensorflow.keras.datasets import cifar10
  (X_train, _), (_, _) = cifar10.load_data()
  X_train = X_train / 127.5 - 1.0  # диапазон [-1, 1]

  # PyTorch
  from torchvision import datasets, transforms
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

  ```
- **Аугментация (опционально для улучшения обобщения):**
  ```
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.ColorJitter(brightness=0.2, contrast=0.2),

  ```
- **Проверка данных:** визуализация 9–16 случайных изображений, проверка распределения классов.

### 2. Реализация архитектур GAN

#### 🔹 Vanilla GAN (базовая)

```
# Генератор: Dense → LeakyReLU → Dense → tanh
# Дискриминатор: Dense → LeakyReLU → Dense → sigmoid

```

- Прост в реализации, но склонен к нестабильности обучения и коллапсу мод.

#### 🔹 DCGAN (рекомендуемая базовая)

| Компонент Рекомендации  |                                                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Генератор**           | `ConvTranspose2d` + `BatchNorm2d` + `ReLU`; вход: `z ~ N(0,1)`, выход: `tanh`                              |
| **Дискриминатор**       | `Conv2d` + `BatchNorm2d` (опционально) + `LeakyReLU(0.2)`; выход: `sigmoid` или `no activation` (для WGAN) |
| **Оптимизатор**         | `Adam(β1=0.5, β2=0.999)`, `lr=2e-4`                                                                        |
| **Loss**                | `BCEWithLogitsLoss` (стабильнее, чем отдельный `sigmoid + BCELoss`)                                        |

#### 🔹 Улучшения для стабильности (опционально)

- **Gradient Penalty** (WGAN-GP): `λ=10`, вычисление градиента на интерполированных сэмплах.
- **Spectral Normalization**: `torch.nn.utils.spectral_norm()` для каждого `Conv`/`Linear` слоя.
- **Two-Time-Scale Update Rule (TTUR)**: разные `lr` для G и D (`lr_D = 4e-4`, `lr_G = 1e-4`).
- **Label Smoothing**: реальные метки `0.9` вместо `1.0` для регуляризации дискриминатора.
- **Progressive Growing** (для CelebA): постепенное увеличение разрешения с 4×4 до 32×32.

### 3. Обучение и мониторинг

- **Цикл обучения:**
  ```
  for epoch in range(num_epochs):
      for real_imgs in dataloader:
          # 1. Обновление дискриминатора
          # 2. Обновление генератора
          # 3. Логирование loss, генерация сэмплов каждые N шагов

  ```
- **Рекомендации:**
  - `batch_size=64–128` для CIFAR-10 (Colab GPU: ≤64 при DCGAN).
  - `num_epochs=100–200` (остановка по визуальному качеству, а не по loss).
  - `mixed_precision` (`torch.cuda.amp` / `tf.keras.mixed_precision`) для ускорения.
  - Сохранение чекпоинтов: `torch.save({'G': G.state_dict(), 'D': D.state_dict()}, ...)`.

### 4. Оценка качества генерации

| Метрика Описание Реализация                |                                                                       |                                                    |
| ------------------------------------------ | --------------------------------------------------------------------- | -------------------------------------------------- |
| **Визуальная оценка**                      | Сетка 8×8 сгенерированных изображений, интерполяция в latent space    | `matplotlib.pyplot`, `torchvision.utils.make_grid` |
| **Inception Score (IS)** ↑                 | Разнообразие + чёткость (только для классифицируемых датасетов)       | `torch-fidelity`, `clean-fid`                      |
| **Fréchet Inception Distance (FID)** ↓     | Расстояние между распределениями реальных и сгенерированных признаков | `clean-fid`, `pytorch-fid`                         |
| **Precision/Recall for Generative Models** | Баланс между качеством и разнообразием                                | `torch-fidelity`                                   |

> 💡 **Важно:** Для FID требуется ≥10,000 реальных и сгенерированных изображений. Используйте `torch-fidelity`:
>
> ```
> pip install clean-fid
> python -m clean_fid.fid --path1 ./real_images --path2 ./fake_images
>
> ```

### 5. Тестирование и визуализация

- **Генерация из случайного шума:**
  ```
  z = torch.randn(64, latent_dim, device=device)
  with torch.no_grad():
      fake = generator(z).cpu()

  ```
- **Интерполяция в latent space:**
  ```
  z1, z2 = torch.randn(2, latent_dim)
  alphas = torch.linspace(0, 1, 10).view(-1, 1)
  z_interp = alphas * z1 + (1 - alphas) * z2

  ```
- **Рекомендация:** Сохраняйте лучшие веса по визуальному качеству, а не по минимальному loss.

---

## Опциональный блок: Сравнение продвинутых архитектур

1. **Реализация одной из архитектур:**
   - **WGAN-GP**: замена loss на Wasserstein distance + gradient penalty.
   - **Conditional GAN**: добавление `class embedding` в G и D для контролируемой генерации.
   - **StyleGAN2-ADA** (упрощённо): использование `StyleBlock`, `AdaIN`, `path length regularization`.
2. **Обучение:**
   - Тоже самое по определению данных, фиксированные `seed`, одинаковое число итераций.
   - Применение `spectral_norm`, `gradient_penalty`, `TTUR` при необходимости.
3. **Сравнение:**
   | Модель FID ↓ IS ↑ Время/эпоху VRAM (ГБ) Параметры (М)  |      |       |         |      |      |
   | ------------------------------------------------------ | ---- | ----- | ------- | ---- | ---- |
   | Vanilla GAN                                            | \~45 | \~5.2 | 45 сек  | 3.1  | 8.2  |
   | DCGAN                                                  | \~32 | \~6.8 | 78 сек  | 4.7  | 12.5 |
   | WGAN-GP                                                | \~28 | \~7.1 | 110 сек | 5.9  | 13.1 |
   | *(опционально) StyleGAN2-ADA*                          | \~15 | \~8.9 | 320 сек | 12.4 | 28.7 |
4. **Анализ:**
   - Почему WGAN-GP стабильнее при обучении?
   - Как `spectral_norm` влияет на градиенты дискриминатора?
   - В чём компромисс между качеством (FID) и вычислительной стоимостью?

---

## 🧩 Опциональные модули (для тех, кто хочет как то улуучшить свой GAN)

- **Аугментация + продвинутая нормализация**: `Albumentations` + `InstanceNorm` / `GroupNorm`
- **WGAN-GP с Gradient Penalty**: реализация `compute_gradient_penalty()` + сравнение стабильности loss
- **Conditional GAN**: генерация по классу + оценка точности классификации сгенерированных изображений
- **Автоматический подбор гиперпараметров**: `Optuna` для `lr`, `β1`, `latent_dim`, `n_critic`
- **Анализ латентного пространства**: PCA/t-SNE проекции `z`, интерполяция, арифметика в latent space
- **Экспорт модели**: `ONNX` / `TorchScript` + инференс-скрипт для генерации из CLI

---

## 📊 Примерный план отчёта (можно использовать этот ноутбук)

# Практическая работа 4: GAN

## 1. Введение

- Цель, постановка задачи, выбранный датасет.

## 2. Описание датасета

- Источник, объём, примеры изображений, распределение классов.

## 3. Методология

### 3.1 Предобработка

- Нормализация, аугментация, DataLoader.

### 3.2 Архитектуры

- Схемы Vanilla GAN / DCGAN / (опционально: WGAN-GP).
- Таблица слоёв с размерностями.

### 3.3 Обучение

- Гиперпараметры, оптимизаторы, расписание learning rate.
- Стратегия стабилизации (gradient clipping, label smoothing).

## 4. Результаты

### 4.1 Динамика обучения

- Графики `loss_G` / `loss_D` по итерациям.

### 4.2 Визуальная оценка

- Сетка сгенерированных изображений (эпохи).
- Интерполяция в latent space.

### 4.3 Количественные метрики

- Таблица: FID, IS, Precision/Recall.
- Сравнение с базовыми моделями.

### 4.4 (Опционально) Сравнение с продвинутой архитектурой

- Таблица сравнения, анализ стабильности, выводы.

## 5. Выводы

- Ограничения подхода, рекомендации по улучшению, практическая применимость.

---

## 🔗 Полезные ссылки и примеры кода

| Ресурс Описание                                                                                                                                                                                              |                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| [DCGAN PyTorch Tutorial](https://www.google.com/url?q=https%3A%2F%2Fpytorch.org%2Ftutorials%2Fbeginner%2Fdcgan_faces_tutorial.html)                                                                          | Официальный туториал с кодом        |
| [WGAN-GP Implementation](https://github.com/caogang/wgan-gp)                                                                                                                                                 | Референсная реализация на PyTorch   |
| [clean-fid](https://github.com/GaParmar/clean-fid)                                                                                                                                                           | Надёжный расчёт FID/IS              |
| [Hugging Face Diffusers](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdocs%2Fdiffusers)                                                                                                       | Для опционального сравнения с DDPM  |
| [Weights & Biases GAN Tutorial](https://www.google.com/url?q=https%3A%2F%2Fwandb.ai%2Fayush-thakur%2Fgan-evaluation%2Freports%2FHow-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI) | Логирование и визуализация обучения |

---

## ⚠️ Примечания по работе в Google Colab

- 🔹 **Фиксируйте случайность**: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`.
- 🔹 **Экономия VRAM**:
  - `batch_size ≤ 64` для DCGAN, `≤ 32` для WGAN-GP.
  - Используйте `mixed_precision` и `gradient_accumulation` для больших моделей.
- 🔹 **Стабильность обучения**:
  - Начинайте с `n_critic=5` (для WGAN), затем уменьшайте до 1.
  - Применяйте `gradient_penalty` с `λ=10` и `eps=1e-6`.
- 🔹 **Сохранение результатов**: регулярно сохраняйте чекпоинты и сэмплы на Google Drive.
- 🔹 **Оценка FID**: требует ≥10,000 изображений — генерируйте батчами, чтобы не переполнить память.

---

📌 *Формат сдачи:* Jupyter Notebook (+ Markdown) или PDF-отчёт + ссылка на воспроизводимый Colab.  