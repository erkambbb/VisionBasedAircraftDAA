# 166 UAV veri seti çalışması (uav_fleet_v3)

## Ne üretir
- `plane_models/` içindeki 166 modelin her biri için 120 kare (`uav_166_tactical_talon` için 1000). Toplam 20 800 kare.
- Her modelde 100 train + 20 valid (talon: 833 + 167).
- Hava çoğunlukla açık (%70 açık, %20 ince bulut, %10 az bulutlu). Ownship 170–1000 ft zeminden yüksekte.
- UAV'ın kameraya en yakın yüzeyi 0.1–300 m arasında. Karelerin %10'u 0.1–5 m, kalanı 5–300 m (log dağılım).
  Çok küçük modellerde 300 m'de uçak 8 pikselden küçük kalırsa üst sınır otomatik düşer (`--min-px 8`).
- Konum: Palo Alto, ownship ±2.5 km (düz zemin). Saat 08:00–17:00 yerel.

## Çıktılar (`datasets/`)
| Klasör | İçerik |
|---|---|
| `uav_fleet_v3/` | Ham X-Plane kareleri (1280x720) + YOLO etiketleri + `state_data.csv` |
| `uav_fleet_v3_cam/` | Kamera görünümlü hali (rpi-cam-style) + etiketler. Eğitimde bu kullanılır. |
| `uav_fleet_v3_cam_boxes/` | Kutuları kırmızı, 1 piksel çizilmiş kontrol görüntüleri |

| `arsiv/` | Eski veri setleri (`uav_fleet_v1*`, `uav_fleet_v2*`, `xp12mm_*`, `sample_mini`, `sample_small`, `pt`) |

Log: `datasets/uav_fleet_v3/fleet_run.log`. İlerleme: `progress.json`.

Klasör renkleri `archive_datasets.sh` ile verilir: `arsiv` turuncu, içinde v1 kum, v2 camgöbeği, xp12mm lacivert, diğerleri gri;
yeni set mavi, `_cam` açık mavi, `_cam_boxes` kırmızı. `sample_small` arşive taşındığı için `src/model/train.py` ve `eval.py`
varsayılan veri seti yerine `-d` ile veri seti verilerek çalıştırılmalı.

## Kutu ve yazı temizliği
- X-Plane UAV'ın yanına `<Anchor deployed>` yazısı çiziyor. `overlay_clean.py` yazıyı bulur ve UAV'sız karenin pikselleriyle siler.
- Yazı ham 1920x1080 karede 1.5 kat büyüktür. Eşleştirme bu yüzden her zaman 1280 genişliğe indirilmiş karede yapılır.
- Eşik 0.6: v2 karelerinde yazısız en yüksek skor 0.57, gerçek yazı en düşük 0.64. Düz gökyüzü (`MIN_ENERGY`) yazı sayılmaz.
- Kutu UAV siluetinin içindeki piksel farkından çıkar. Gölge ve yazı kutuyu genişletmez.
- Kare ancak kutunun dört kenarı da geometrik kutuyla örtüşürse kaydedilir: eksen başına en fazla 3 piksel ya da kutu boyunun %4'ü fark.
  Uymazsa (`mismatch`) kare atılır. Böylece kutu ne geniş kalır ne de uçağın bir parçasını dışarıda bırakır.
- İki kare farklı sahne gösteriyorsa (UAV dışında karenin %2'sinden fazlası değişmişse, `unstable`) kare atılır.
- Yazı UAV üstündeyse (`label`) kare atılır.

## Model dosyaları (bir kez)
`plane_models/` içindeki `.acf` dosyaları X-Plane'in istediği alanlardan yoksun ("Vne of zero" hatasıyla X-Plane kapanıyor).
`python3 make_fleet_acfs.py` her model için tam bir `.acf` (shadow_twin_boom şablonu + modelin `.obj` dosyası) üretip
`X-Plane 11/Aircraft/Extra Aircraft/uav_fleet/` altına yazar.

## Gece öncesi (bir kez)
1. X-Plane'i aç, Cessna 172SP ile uçuşa başla. Flight Configuration'da **1 AI uçak** seçili olsun (hangisi olduğu önemli değil, script değiştirir).
2. X-Plane tam ekran monitör 1'de, önünde başka pencere yok. Terminali başka çalışma alanına al.
3. Ekran kilidini ve uykuyu kapat (script `systemd-inhibit` ve `xset` ile de dener).
4. Prize tak. Laptop ise havalandırma açık dursun.
5. Deneme: `cd src/data_generation && python3 -m generate_fleet --smoke` (3 model x 4 kare, ~1 dk).
   Sonra `python3 draw_boxes.py uav_smoke` ve `datasets/uav_smoke_boxes/` içine bak: kutu uçağı sarıyor mu?

## Başlatma
X-Plane açık ve uçağın içinde olmalı (Palo Alto, mtab penceresi kapalı).
```bash
cd src/data_generation
./run_fleet_night.sh uav_fleet_v3
```
Komut önce eski veri setlerini `datasets/arsiv/` altına taşır, sonra üretimi başlatır. Bitince kamera görünümü (rpi-cam-style) ve
kırmızı 1 piksel kutular otomatik üretilir, kamera kadrajı dışında kalan kareler silinir. Yarıda kesilirse aynı komut kaldığı modelden devam eder.

## 15 dakikalık kontrol
- Üretim kendi kendini 15 dakikada bir denetler: son 12 kareyi okur, `datasets/uav_fleet_v3/kontrol/son_kontrol.jpg` dosyasını yazar.
  3 ya da daha çok karede yazı kalmışsa, görüntü düz renkse ya da kutu çerçeve dışındaysa durur. Tek şüpheli kare yalnız loga yazılır.
- Dışarıdan durum: `python3 check_run.py uav_fleet_v3`. Son satır `DURUM: tamam` ya da `DURUM: SORUN (...)` olur.
  Süreç, X-Plane, kare sayıları, KONTROL satırları ve 5 dakikadır yeni kare gelmemesi denetlenir.

## Hız
- Sabit bekleme, ısı beklemesi ve mola yok. Sahne değişince yalnız birkaç render karesi beklenir (3–30 kare, 0.08–1 sn).
  Süre X-Plane'in o anki kare süresinden hesaplanır; kötü kare gelirse otomatik artar, iyi karelerde azalır.
- Pencere görüntüsü doğrudan X sunucusundan alınır (`python-xlib`), geometri tek XPlaneConnect isteğiyle okunur.
- Hava 20 karede bir değişir (1 sn), model yükleme beklemesi 3 sn.
- Kamera dönüşümü 6 süreçte paralel çalışır (`--cam-workers`).

## Sorun olursa
- `UavSwitch cevap vermedi`: eklenti yüklenmemiş. X-Plane'i yeniden başlat, `Log.txt` içinde `UavSwitch` ara.
- `UAV ekranda görünmüyor`: model yüklenmemiş ya da ekranı başka pencere kapatıyor. Script modeli bir kez daha dener, olmazsa `progress.json` içinde `failed` listesine yazar.
- X-Plane kapanırsa: yeniden aç, eski python sürecini PID ile durdur, aynı komutu çalıştır.
- Eklenti yeniden derleme: `cd src/xplane_plugin/uav_switch && make install`.
