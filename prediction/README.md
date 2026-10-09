# חיזוי עם BridgePulse

הפייפליין קורא הקלטה אחת מה־H5, מחלץ פיצרים וטוען מודל מוכן. הוא אינו מאמן מודל ואינו קורא את התווית.

מתיקיית הפרויקט:

```powershell
.venv\Scripts\python.exe -m prediction --h5 data/raw/Rt345Bridge.h5 --recording-path damage_1/acceleration/test_1 --output outputs/predictions/example.json
```

במחברת:

```python
from prediction.pipeline import BridgePredictor
predictor = BridgePredictor()
result = predictor.predict_h5(file_path, "damage_1/acceleration/test_1")
result
```

אם המחברת פתוחה מתוך `notebooks`, הוסף קודם את תיקיית הפרויקט ל־`sys.path`, או השתמש בהגדרת ROOT שכבר קיימת במחברת. לא הוספנו תאי קוד חדשים למחברת לצורך החיזוי.

להקלטה חדשה, אפשר לקרוא `predict_signals(signals, fs=128)` עם מילון של 58 ערוצים, שמפתחותיו הם `(sensor, axis)`, למשל `("A01", "y")`. כך אין צורך לשייך אותה מראש לקבוצת נזק או בסיס.

הקלט דורש חיישנים A01–A30 למעט A14, צירים y/z, קצב 128Hz ולפחות 89 שניות. משתמשים ב־89 השניות הראשונות בלבד. אין מילוי של חיישנים חסרים או נרמול שמותאם להקלטה. ה־H5 נשאר ללא שינוי.

הפלט מכיל תווית 0=בסיס או 1=נזק, וציוני מודל. הציונים אינם הסתברויות מכוילות. המודל הוא מועמד מחקרי לזיהוי מצב קיים, לא חיזוי הופעת סדקים בעתיד.

## הקבצים

- `pipeline.py`: קריאת H5 וחיזוי מאותות.
- `bridge_model.py`: חוזה הקלט וחילוץ MFCC.
- `artifacts/model.joblib`: המודל שאומן על כל 25 ההקלטות.
- `artifacts/manifest.json`: פרמטרים, פיצרים, גרסאות וחתימות.
- `artifacts/model_card.md`: מגבלות והערכת המודל.
- `requirements.txt`: תלויות לחיזוי בלבד.

זהו המודל שאומן על כל הדאטה, ולכן חיזוי ההקלטות המקוריות אינו הערכת Holdout. מודל בדיקת ה־Holdout נשאר בנפרד ב־`models/bridge_damage_rf_group_holdout.joblib`. קבצי המחקר נשארו ב־`scripts`, המחברת ב־`notebooks`, ותוצאות הניסויים ב־`outputs`.
