# AlarmGuard (Data Science + Computer Vision layihəsi)

Bu layihə sənin dediyin ideyanın **tam texniki skeletini** verir:

1. Kamera ilə real vaxt monitorinqi.
2. Pomodoro timer (fokus/fasilə rejimi).
3. Gözlər 5+ saniyə bağlı qalanda alarm video açılması.
4. Əlavə mini fun-game: üz ifadəsinə görə meymun emojisi.
5. Modellərin scratch-dan train olunması, save və runtime-da yüklənməsi.

## Arxitektura

- `scripts/train_eye_model.py` → göz açıq/qapalı CNN modeli (`models/eye_state_cnn.keras`).
- `scripts/train_expression_model.py` → expression CNN modeli (`models/expression_cnn.keras`).
- `src/alarm_guard/vision.py` → OpenCV + MediaPipe ilə göz bölgəsinin çıxarılması və model inferensi.
- `src/alarm_guard/pomodoro.py` → Pomodoro engine.
- `src/alarm_guard/alarm.py` → VLC ilə alarm video full screen + yüksək səs.
- `src/alarm_guard/fun_game.py` → expression nəticəsini monkey emoji-ə çevirir.
- `src/alarm_guard/app.py` → bütün modulların birləşdirilməsi.

## Quraşdırma

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset strukturu

### 1) Göz modeli (binary)

```text
data/eye_state/
  train/
    open/
    closed/
  val/
    open/
    closed/
```

### 2) Expression modeli (4 sinif)

```text
data/expressions/
  train/
    angry/
    happy/
    sad/
    surprised/
  val/
    angry/
    happy/
    sad/
    surprised/
```

## Train

```bash
python scripts/train_eye_model.py
python scripts/train_expression_model.py
```

Model faylları `models/` altına yazılır.

## App-i işə salmaq

```bash
PYTHONPATH=src python src/alarm_guard/app.py --config config.yaml
```

Custom alarm URL/fayl vermək üçün:

```bash
PYTHONPATH=src python src/alarm_guard/app.py --config config.yaml --alarm "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

## Klavişlər

- `g` → mini monkey game prediction.
- `s` → alarm stop.
- `q` → çıxış.

## Data Science tövsiyələri (layihəni “mükəmməl” etmək üçün)

- Train/val/test split sabit random seed ilə apar.
- Class imbalance varsa weighted loss və ya oversampling et.
- Eye modeldə ROC-AUC və recall-a fokuslan (false-negative az olsun).
- Təhlükəsizlik üçün `eyes_closed_seconds_threshold` şəxsi kalibrasiyaya görə 3-7 aralığında test et.
- Domain shift (gecə işığı, eynək, kamera keyfiyyəti) üçün augmentations əlavə et.
- `DVC` və ya MLflow ilə model versiyalama et.

## Mühüm qeydlər

- Sistem səssiz (mute) halında OS səviyyəsində səsi 100% “zorla açmaq” platformadan asılıdır.
  Bu repo player səviyyəsində audio volume-u maksimum verir. İstəsən növbəti addımda OS-specific modul (Windows/macOS/Linux ayrı) əlavə edə bilərik.
- YouTube URL-lərin VLC ilə oynadılması üçün bəzi sistemlərdə əlavə codec və ya youtube-dl backend lazım ola bilər.

