# Eğitim öncesi hazırlık ve augmentasyon (YOLO26, RF-DETR)

İki model aynı veriyle, aynı split'lerle ve aynı çözünürlükte eğitilir. En iyisi gerçek test setinde seçilir.

## 1. Veri seti bileşimi

| Kaynak | Kare | UAV var mı | Not |
|---|---|---|---|
| Sentetik UAV (`uav_fleet_v4_cam`) | 20 000 | var | 16 000 tek UAV, 2000 iki UAV, 2000 üç-dört UAV; %60 kırsal (%40 Türkiye), %40 şehir |
| Sentetik arka plan (`bg_*`) | 1000 | yok | UAV karesi gibi kurulur, UAV gösterilmez |
| Gerçek UAV | 30 000 | var | |
| Gerçek arka plan | 2000 | yok | Mümkünse yarışma kamerası ve yarışma alanı |
| Gökyüzü | 1000 | yok | Sadece gökyüzü ve bulut |
| COCO (airplane yok) | 1000 | yok | |
| Skyview (yere dik bakış) | 1000 | yok | |
| **Toplam** | **56 000** | | Negatif 6000 (%10.7) |

Tek sınıf vardır: `0 = uav`. Bütün kaynaklarda sınıf numarası 0 olmalı.

## 2. Eğitimden önce yapılacaklar

### Split (en önemli adım)
- **Sentetik:** Split hazır gelir. `fleet_core.valid_units` bir bloğun bütün karelerini tek tarafa koyar. Valid ~3480 kare.
- **Gerçek UAV:** Split'i video ya da uçuş bazında yap. Bir videonun kareleri komşu karelere çok benzer. Bir kısmı train'de, bir kısmı valid'de kalırsa valid skoru şişer.
- **Ardışık kareleri seyrelt:** Videodan alınan karelerde art arda neredeyse aynı kareler olur. Her N karede bir kare al ya da perceptual hash ile benzerleri ayıkla.
- **Ayrı bir gerçek test seti tut:** Eğitimde hiç kullanılmamış uçuşlardan oluşsun. Model seçimi bu setle yapılır, çünkü yarışmaya en yakın ölçüm budur. Sentetik valid skorunu ayrı raporla.
- **Arka planlar:** Her arka plan kaynağında da 1/6 valid ayır. Böylece yanlış pozitif oranı valid'de de ölçülür.

### Arka plan kaynakları
- **COCO:** `train2017` içinden `airplane` (kategori id 5) etiketi olan görüntüleri tamamen çıkar. Uçağı silmeye ya da boyamaya (inpaint) gerek yok.
  - COCO'da küçük ve etiketlenmemiş uçaklar olabilir. Kalan görüntüleri COCO ile önceden eğitilmiş bir detektörden geçir, `airplane` skoru 0.3'ün üstünde olanları at.
  - Rastgele seçim yerine dış mekan ve gökyüzü içeren görüntüleri tercih et. `bird` ve `kite` içerenler gökyüzünde küçük nesne olduğu için en faydalı negatiflerdir.
- **Skyview:** 15 sınıf × 800 görüntü, her biri 256×256 (AID ve NWPU-RESISC45 kaynaklı).
  - `airport` sınıfını alma: apron fotoğraflarında yerde uçak var.
  - Görüntüleri 1280'e büyütme. 5 kat büyütülmüş bulanık görüntü modele "bulanık = arka plan" kısayolunu öğretir.
  - Bunun yerine 5×3 kolaj yap: 5 × 256 = 1280 genişlik, 3 × 256 = 768 yükseklik, sonra 720'ye kırp. Böylece her karo kendi çözünürlüğünde kalır. 1000 kolaj için ~15 000 karo gerekir ve 14 sınıfta 11 200 görüntü var. Bu yüzden ya ~670 kolaj yap ya da karoları farklı kırpma ve döndürmeyle tekrar kullan.
- **Gökyüzü:** Açık hava, parçalı bulut, kapalı hava, pus ve güneşe karşı çekim karışık olsun. İçinde kuş ya da uçak olmadığını gözle kontrol et.
- **Gerçek arka plan:** En değerli negatifler bunlar. Yarışma kamerasıyla çekilmiş, UAV'sız kareler olmalı: yarışma alanının zemini, ufku ve bulutu.

### Görüntü ve etiket kontrolü
- Sentetik için `_cam` sürümünü kullan, ham sürümü değil. Hareket bulanıklığı ve kamera taklidi `_cam`'de. Gerçek fotoğraflara camstyle uygulanmaz.
- Negatif görüntülerde label dosyası boş olsun ya da hiç olmasın. İki model de bunu "nesne yok" olarak okur.
- Gerçek etiketlerde 1280 genişlikte 8 pikselin (640 girişte 4 px) altındaki kutuları gözden geçir. Bunlar ya etiketleme hatasıdır ya da hiçbir model bulamaz.
- Her kaynaktan 50 görüntüyü kutularla çizip gözle kontrol et (`draw_boxes.py`).

### Çözünürlük ve donanım
- **Donanım:** T3 Gemstone (TI AM67A), RPi HQ kamera (IMX477R, 12 mm), en az 15 FPS. Kaptanın ölçümü 18-19 FPS, ama hangi giriş boyutunda olduğu bilinmiyor.
- **Önce FPS'i ölç:** Her iki model için Gemstone'da 640, 800 ve 960 girişle ölç. 15 FPS'i sağlayan en büyük boyutu seç.
- **Eğitim ve inference aynı boyutta:** Seçilen boyut eğitimde de kullanılır (imgsz / resolution).
- **Küçük UAV:** Veri 1280×720. Sentetik setteki en küçük UAV 1280 genişlikte 12 piksel, yani 640 girişte 6 piksel. Uzak sınır 220 m: 1,7 m'lik bir UAV 640'ta ~9 piksel.
- **Kaçarsa:** 640'ta küçük UAV kaçıyorsa giriş boyutunu 800-960'a çıkar ya da P2 başlığını dene.
- **YOLO26:** P2 başlığı `yolo26s-p2.yaml`. P2'nin hazır ağırlığı yok, omurga `YOLO("yolo26s-p2.yaml").load("yolo26s.pt")` ile yüklenir. P2 başlığı FPS'i düşürür, Gemstone'da ölç.
- **RF-DETR:** `resolution`, `patch_size × num_windows` değerinin katı olmalı (dokümana göre 32'nin katı: 640, 800, 960).
  - RF-DETR deformable attention kullanır. TI'nin derleyicisi (TIDL) bu katmanları hızlandıramazsa iş ARM işlemciye düşer ve FPS çok düşer.
  - Kaptanın 18-19 FPS'i hangi modelde alındı, netleştir. RF-DETR'i Gemstone'a derleyip FPS'i karşılaştırmadan seçme.

### Format
- İki model de YOLO formatını okur: `data.yaml` + `images/train|valid` + `labels/train|valid`. RF-DETR YOLO klasörünü otomatik tanır.
- Bütün kaynakları tek bir `data.yaml` altında topla. Train ve valid listelerini dosya yolu listesi (`.txt`) olarak vermek en kolayı.

## 3. Eğitim sırasında augmentasyon

Genel kural: Gökyüzü yukarıda, zemin aşağıdadır. Dikey çevirme yapılmaz.

Küçük UAV'ı bozan augmentasyonlar düşük tutulur: güçlü küçültme, mixup, büyük bulanıklık.

### YOLO26 (Ultralytics)
```python
from ultralytics import YOLO
model = YOLO("yolo26s.pt")            # ya da yolo26m.pt; küçük nesne için yolo26s-p2.yaml denemesi
model.train(
    data="all.yaml", imgsz=640, epochs=150, batch=-1,    # Gemstone'da seçilen boyut (640/800/960)
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,      # renk ve parlaklık: hava ve saat farkları
    degrees=10.0,                           # küçük döndürme: kamera roll'u
    translate=0.1, scale=0.4,               # 0.5'ten fazla küçültme 8 px UAV'ı yok eder
    shear=0.0, perspective=0.0,
    fliplr=0.5, flipud=0.0,                 # yatay çevirme var, dikey yok
    mosaic=1.0, close_mosaic=15,            # son 15 epoch mozaiksiz: gerçek kare düzeni
    mixup=0.0, copy_paste=0.0,              # yarı saydam hayalet UAV yok; copy_paste maske ister
)
```
- `albumentations` paketi kuruluysa Ultralytics kendi düşük olasılıklı setini (Blur, MedianBlur, ToGray, CLAHE) otomatik ekler.
- Sentetik karelerde hareket bulanıklığı ve JPEG zaten `_cam` içinde var. Ek bulanıklık gerekmez.

### RF-DETR
```python
from rfdetr import RFDETRMedium            # Nano / Small / Medium / Base / Large
model = RFDETRMedium()
aug = {
    "HorizontalFlip": {"p": 0.5},
    "Rotate": {"limit": 10, "p": 0.3},
    "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5},
    "HueSaturationValue": {"hue_shift_limit": 5, "sat_shift_limit": 25, "val_shift_limit": 20, "p": 0.4},
    "MotionBlur": {"blur_limit": (3, 5), "p": 0.15},
    "GaussNoise": {"std_range": (0.01, 0.04), "p": 0.2},
    "ImageCompression": {"quality_range": (55, 95), "p": 0.3},
}
model.train(dataset_dir="datasets/all", epochs=60, batch_size=4, grad_accum_steps=4,
            resolution=640, aug_config=aug)           # Gemstone'da seçilen boyut (640/800/960)
```
- Kurulum: `pip install "rfdetr[train,augment]"`. Hazır presetler `rfdetr.datasets.aug_configs` altında (1.8.0'dan önceki sürümlerde modülün adı tekil: `aug_config`).
- `AUG_AERIAL` kullanma. Yukarıdan bakış içindir, dikey çevirme ve büyük döndürme yapar.
- `aug_config` verilmezse varsayılan olarak ölçek oynatma ve %50 yatay çevirme uygulanır. `{}` verilirse çevirme de kapanır.
- `MotionBlur`, `GaussNoise` ve `ImageCompression` gerçek kareleri de kameraya benzetir. Olasılıkları düşük tutulur, çünkü sentetik kareler bunları zaten camstyle'dan alıyor.

## 4. Karşılaştırma ve seçim

Aynı split'ler, aynı çözünürlük ve aynı gerçek test seti kullanılır. Şunlara bakılır:
- **mAP50 ve recall, boy gruplarına göre:** küçük (<16 px), orta (16–64 px), büyük (>64 px). Uzak UAV'ı kim daha iyi buluyor?
- **Yanlış pozitif:** arka plan setlerinde görüntü başına yanlış kutu, kaynak kaynak (bulut, şehir, COCO, Skyview).
- **Ayrı skorlar:** sentetik valid ve gerçek valid. Aradaki fark büyükse sim-to-real açığı vardır.
- **FPS:** yarışma donanımında, seçilen çözünürlükte.
- **Talon:** talon karelerinde ayrı recall.

İsteğe bağlı ikinci aşama: önce tüm veriyle eğit, sonra birkaç epoch gerçek veri ve biraz sentetikle düşük lr'de fine-tune yap.
