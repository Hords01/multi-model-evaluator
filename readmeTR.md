# 🧠 Göğüs Röntgen Görüntüleri için Çoklu Model Sınıflandırma

Bu depo, **dört farklı sınıftan** oluşan bir göğüs röntgeni veri kümesi üzerinde birden fazla derin öğrenme modelini karşılaştırmak için modüler ve genişletilebilir bir eğitim altyapısı sunar. Temel amaç, çeşitli CNN mimarilerini doğruluk, eğitim verimliliği ve tanısal performans açısından değerlendirmektir.

## 📌 Özellikler

- ✅ `torchvision.models` üzerinden 27+ CNN mimarisi desteği:
  - ResNet (18, 34, 50, 101, 152)
  - DenseNet (121, 161, 169, 201)
  - VGG (11, 13, 16, 19 - batch norm'lu/olmadan)
  - MobileNetV2, MobileNetV3 (Large/Small)
  - EfficientNet (B0, B1)
  - AlexNet, SqueezeNet (1.0, 1.1)
  - ShuffleNetV2 (x0.5, x1.0)
  - GoogLeNet, InceptionV3 (yardımcı başlıklarla birlikte)
- 🏥 Göğüs röntgeni görüntülerini dört sınıfa ayırır (örn: Normal, Zatürre, COVID-19, Tüberküloz)
- 📊 Temel değerlendirme metriklerini izler:
  - Doğruluk (Accuracy)
  - Kesinlik (Precision)
  - Duyarlılık (Recall)
  - F1 Skoru
  - Karmaşıklık Matrisi (Confusion Matrix)
- 💾 Şunları kaydeder:
  - Model ağırlıkları (`.pt`)
  - Eğitim geçmişi (CSV formatında)
  - Karmaşıklık matrisi görselleri
  - Modeller arası nihai karşılaştırma özeti
- ⚙️ Inception ve GoogLeNet gibi özel mimarileri destekler (yardımcı çıktılar dahil)
- 📈 Eğitim/doğrulama metriklerinin grafiklerini otomatik olarak oluşturur

## 📂 Veri Kümesi

Bu proje için kullanılan veri kümesi Kaggle'da yer almaktadır:

> 🔗 [Chest X-ray Pneumonia, COVID-19, Tuberculosis Dataset – Kaggle](https://www.kaggle.com/)

### Klasör Yapısı

Eğitime başlamadan önce veri kümesini şu şekilde yapılandırın:
```
data/
├── train/
│ ├── Normal/
│ ├── Pneumonia/
│ ├── COVID-19/
│ └── Tuberculosis/
├── val/
│ ├── Normal/
│ └── ...
└── test/
├── Normal/
└── ...
```

Her alt klasör, o sınıfa ait görüntüleri içermelidir. Bu yapı, PyTorch’un `ImageFolder` sınıfı ile uyumludur.

---

### 📊 Çıktılar

Eğitim tamamlandıktan sonra `results/` klasörü içinde aşağıdaki dosyalar oluşturulur:
```
results/
├── resnet18_history.csv # Eğitim ve doğrulama metrikleri
├── resnet18_confusion_matrix.png # Karmaşıklık matrisi görseli
├── resnet18.pt # Eğitilmiş model ağırlıkları
├── ...
summary.csv # Tüm modeller arası nihai karşılaştırma özeti
```

---

### 🧪 Değerlendirme Metrikleri

Tüm modeller aşağıdaki metriklerle değerlendirilir:

- **Doğruluk (Accuracy)**
- **Kesinlik (Precision)**
- **Duyarlılık (Recall)**
- **F1 Skoru**
- **Karmaşıklık Matrisi (Confusion Matrix)**

---

### 🧾 Atıf / Kaynak Belirtme

Bu depoyu, içindeki kodları veya herhangi bir bölümünü araştırmanızda, yayında ya da projede kullanırsanız lütfen atıf yapınız veya uygun bir şekilde kaynak belirtiniz:

**Emirkan Beyaz**, "Multi-Model Evaluator", GitHub Repository, 2025.  
🔗 https://github.com/Hords01/multi-model-evaluator

---

### ✉️ İletişim

Sorularınız, hatalar, ya da iş birliği teklifleri için GitHub üzerinden Issue veya Discussion açabilirsiniz.  
📧 E-posta: **beyazemirkan@gmail.com**
