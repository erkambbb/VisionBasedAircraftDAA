# 166 UAV veri seti çalışması (uav_fleet_v4)

## Ne üretir
- Toplam 22 000 kare. 16 000 tek UAV: 165 model x 90 kare, `uav_166_tactical_talon` 1150 kare.
- 2000 kare 2 UAV'lı (100 grup x 20), 2000 kare 3-4 UAV'lı (100 grup x 20). Grupların yarısında talon var.
  Her UAV kendi mesafesini ayrı seçer (biri 100 m, öbürü 20 m gibi). Resimde üst üste binen UAV'lar atılır.
- 2000 kare uçaksız arka plan (100 grup x 20, `bg_*`, boş etiket dosyası). Model arka planı öğrensin diye.
- Her model, grup ve arka plan grubu aynı karışımı alır (`fleet_core.unit_plan`): hava payları, mesafe bantları,
  şehir türü, aşağı bakış, saat ve irtifa her birimde tüm aralığa eşit dağılır. Hiçbir model tek havada kalmaz.
  Hava her birimde bloklar halinde (20 karelik grupta 8 açık, 2 ince bulut, 1'er bulutlu/yağmur/fırtına/sis, 6 karışık).
- Her model ve grupta yaklaşık 5/6 train, 1/6 valid.
- Mesafe (kameraya en yakın yüzey): %10 0.25–25 m (log), %50 25–180 m, %40 180–300 m.
  Küçük modelde 300 m'de uçak 8 pikselden küçük kalırsa üst sınır düşer (`--min-px 8`).
- Hava (20 karede bir): %40 açık, %10 ince bulut, %5 bulutlu, %5 yağmur, %5 fırtına, %5 sis, %30 karışık
  (rastgele katmanlar, pus, hafif yağmur). Bulut tabanı her zaman ownship'in üstünde.
- Ownship zeminden 150–1000 ft yüksekte. Karelerin %45'inde kamera aşağı bakar (-8 ile -35 derece), UAV şehrin önünde görünür.
- Konum: dünyada 17 şehir (`fleet_core.CITIES`, `fleet_core.SPOTS`, toplam 163 nokta): New York, Chicago, Washington DC,
  Toronto, Miami, Las Vegas, Los Angeles, Seattle, London, Paris, Frankfurt, Barcelona, Innsbruck, İstanbul, Dubai,
  Tokyo, Sydney. Her şehirde gökdelen/yüksek blok, orta katlı ve alçak ev noktaları (%45/%25/%30).
  Bina arkasında kalan UAV'lı kare kutu tutmadığı için atılır.
- Her modelin 90 karesi 30'luk 3 bloğa bölünür, her blok başka şehirde. Grup ve arka plan setleri (20 kare) tek şehir.
  Şehirler sırayla dağıtılır (her şehre ~1290 kare); çalışma şehir şehir gider, her şehre bir kez geçilir
  (`fleet_core.plan_units`). Şehre geçişte sahnenin yüklenmesi beklenir (en çok 240 sn + 15 sn).
- Havalimanları X-Plane'in `apt.dat` dosyasından gelir (`make_city_airports.py` -> `city_airports.py`):
  pisti 1500 m ve üstü olanlara 4 km, küçüklere 1,5 km'den yakın uçulmaz; yerde uçak görünmesin.
  X-Plane'de park halindeki uçaklar zaten kapalı (`renopt_static_acf 0`).
- Saat her şehrin kendi yerel saati (yaz saati dahil); Sydney'de yaz Aralık–Mart.
- Zemin yüksekliği UavSwitch eklentisinin arazi yoklamasından gelir (şehir başına bir kez, `terrain_<şehir>.npz`).
- Ham kareler JPEG kalite 100. Kamera görünümü (`_cam`) gerçek kamera taklidi olarak JPEG 70 kalır.
- Saat 08:00–18:00 yerel (yaz saati), tarih hep yaz (21 Haziran – 22 Eylül).

## Çıktılar (`datasets/`)
| Klasör | İçerik |
|---|---|
| `uav_fleet_v4/` | Ham X-Plane kareleri (1280x720) + YOLO etiketleri + `state_data.csv` |
| `uav_fleet_v4_cam/` | Kamera görünümlü hali (rpi-cam-style) + etiketler. Eğitimde bu kullanılır. |
| `uav_fleet_v4_cam_boxes/` | Kutuları kırmızı, 1 piksel çizilmiş kontrol görüntüleri |

| `arsiv/` | Eski veri setleri (`uav_fleet_v1*`, `uav_fleet_v2*`, `uav_fleet_v3*`, `xp12mm_*`, `sample_mini`, `sample_small`, `pt`) |

Log: `datasets/uav_fleet_v4/fleet_run.log`. İlerleme: `progress.json`.

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
- Kadraj kenarında kesilen uçaklar silinmez. Karelerin yaklaşık %15'inde uçak bilerek kenara konur (kareye giren ya da çıkan uçak).
  Kutu yalnız kameraya giren uçak piksellerinden çıkar, görünen parçayı sarar. Ham setin etiketi de bu görünen kutudur.
- Kameraya hiç girmeyen ya da 3 pikselden az görünen uçakta kare atılır (`outside`).

## Model dosyaları (bir kez)
`plane_models/` içindeki `.acf` dosyaları X-Plane'in istediği alanlardan yoksun ("Vne of zero" hatasıyla X-Plane kapanıyor).
`python3 make_fleet_acfs.py` her model için tam bir `.acf` (shadow_twin_boom şablonu + modelin `.obj` dosyası) üretip
`X-Plane 11/Aircraft/Extra Aircraft/uav_fleet/` altına yazar.

## Gece öncesi (bir kez)
1. X-Plane'i aç, herhangi bir yerde Cessna 172SP ile uçuşa başla (script şehirlere kendisi gider). Flight Configuration'da **4 AI uçak** seçili olsun (hangileri olduğu önemli değil, script değiştirir).
   UavSwitch eklentisi yeni (slot destekli) sürüme güncellendi: X-Plane'i yeniden başlat.
2. X-Plane tam ekran monitör 1'de, önünde başka pencere yok. Terminali başka çalışma alanına al.
3. Ekran kilidini ve uykuyu kapat (script `systemd-inhibit` ve `xset` ile de dener).
4. Prize tak. Laptop ise havalandırma açık dursun.
5. Deneme: `cd src/data_generation && python3 -m generate_fleet --smoke` (3 model x 4 kare, ~1 dk).
   Deneme 2 çoklu grubu da içerir (2 ve 3 UAV x 4 kare). Sonra `python3 draw_boxes.py uav_smoke` ve `datasets/uav_smoke_boxes/` içine bak: kutu uçağı sarıyor mu?

## Başlatma
X-Plane açık ve uçağın içinde olmalı (mtab penceresi kapalı).
```bash
cd src/data_generation
./run_fleet_night.sh uav_fleet_v4
```
Komut önce eski veri setlerini `datasets/arsiv/` altına taşır, sonra üretimi başlatır. Bitince kamera görünümü (rpi-cam-style) ve
kırmızı 1 piksel kutular otomatik üretilir, kamera kadrajı dışında kalan kareler silinir. Yarıda kesilirse aynı komut kaldığı modelden devam eder.

## 15 dakikalık kontrol
- Üretim kendi kendini 15 dakikada bir denetler: son 12 kareyi okur, `datasets/uav_fleet_v4/kontrol/son_kontrol.jpg` dosyasını yazar.
  3 ya da daha çok karede yazı kalmışsa, görüntü düz renkse ya da kutu çerçeve dışındaysa durur. Tek şüpheli kare yalnız loga yazılır.
- Dışarıdan durum: `python3 check_run.py uav_fleet_v4`. Son satır `DURUM: tamam` ya da `DURUM: SORUN (...)` olur.
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
