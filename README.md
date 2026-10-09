# BridgePulse

BridgePulse הוא פרויקט מחקר לזיהוי מצב נזק ניסויי בגשר מתוך הקלטות של חיישני תאוצה. הפייפליין קורא הקלטה אחת, מחלץ מאפייני אות באמצעות MFCC ומפעיל מודל Random Forest מוכן כדי לסווג אותה כמצב בסיס (`baseline`) או נזק (`damage`).

המודל מסווג את מצב ההקלטה; הוא אינו מנבא הופעת סדקים בעתיד. זהו מועמד מחקרי, ולא מערכת מאומתת לקבלת החלטות בטיחות בגשרים.

## התחלה מהירה: חיזוי

הפקודות הבאות מיועדות ל־Windows PowerShell, מתוך תיקיית הפרויקט. הסביבה הקיימת נבדקה עם Python 3.12.3.

### 1. הורדת הפרויקט והתקנת תלויות

```powershell
git clone https://github.com/amitnahum18/BridgePulse.git
cd BridgePulse
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r prediction/requirements.txt
```

התקנה זו מספיקה לחיזוי עם המודל המצורף. אין צורך לאמן אותו מחדש או להתקין MLflow. לעבודה עם המחברת והניסויים, ראו את סעיף סביבת המחקר בהמשך.

### 2. הוספת קובץ הנתונים

קובץ המקור `Rt345Bridge.h5` אינו כלול במאגר בשל גודלו, כ־907MB. יש להשיג אותו בנפרד ולהציב אותו בנתיב הבא לפני הרצת הדוגמה או המחברת:

```text
data/raw/Rt345Bridge.h5
```

חתימת SHA-256 של הקובץ המקורי נמצאת ב־[data/raw/Rt345Bridge.h5.sha256](data/raw/Rt345Bridge.h5.sha256). ניתן לבדוק את החתימה המקומית באמצעות:

```powershell
Get-FileHash -Algorithm SHA256 data/raw/Rt345Bridge.h5
```

אפשר להשתמש בקובץ H5 אחר העומד בדרישות הקלט ולהעביר את הנתיב שלו באמצעות `--h5`.

### 3. הרצת חיזוי

```powershell
.\.venv\Scripts\python.exe -m prediction --h5 data/raw/Rt345Bridge.h5 --recording-path damage_1/acceleration/test_1 --output outputs/predictions/example.json
```

`--recording-path` הוא הנתיב לקבוצת החיישנים בתוך קובץ ה־H5. שם הקבוצה משמש לאיתור המדידות בלבד ואינו משמש כמאפיין או כתווית בחיזוי. קובץ ה־H5 נפתח לקריאה בלבד.

התוצאה מודפסת כ־JSON ונשמרת בנתיב שניתן ב־`--output`. אפשר להשמיט את `--output` כדי להדפיס בלבד.

| שדה בפלט | משמעות |
|---|---|
| `predicted_label` | `0` = מצב בסיס, `1` = נזק ניסויי |
| `predicted_class` | `baseline` או `damage` |
| `damage_score`, `baseline_score` | ציוני המודל לשתי המחלקות; אינם הסתברויות מכוילות |
| `status` | `exploratory_candidate` — מועמד מחקרי |

להוראות שימוש ב־Python ובמחברת, ראו [prediction/README.md](prediction/README.md).

## דרישות הקלט

| רכיב | דרישה |
|---|---|
| חיישנים | `A01` עד `A30`, למעט `A14`: סך הכול 29 חיישנים |
| צירים | `y` ו־`z` לכל חיישן: סך הכול 58 ערוצים |
| קצב דגימה | 128Hz |
| משך | לפחות 89 שניות לכל ערוץ; החיזוי משתמש ב־89 השניות הראשונות |
| ערכי האות | מערכים חד־ממדיים של ערכים סופיים |
| מבנה H5 | קבוצת הקלטה המכילה קבוצות חיישנים; בכל חיישן מערכי `y` ו־`z` ומאפיין `sample_rate=128` |

חיישנים חסרים, אותות קצרים או ערכים לא תקינים גורמים לשגיאה. אין השלמה אוטומטית של ערוצים חסרים. אפשר גם להעביר אותות ישירות ל־`BridgePredictor.predict_signals`; דוגמה נמצאת בתיעוד החיזוי.

## המודל וההערכה

המודל המצורף הוא Random Forest עם 200 עצים, עומק מרבי 2 ומינימום 4 דגימות בעלה. חילוץ המאפיינים משתמש בחלונות של 4 שניות בצעד של 2 שניות. מכל ערוץ מחושבים 8 מקדמי MFCC, המסוכמים באמצעות חציון, טווח בין־רבעוני ושינוי מוחלט ממוצע בין חלונות: בסך הכול 1,392 מאפיינים להקלטה.

ההערכה שומרת את כל ההקלטות של אותו מצב ניסויי יחד, כך שהמצב אינו משותף לקבוצות האימון והוולידציה באותה חלוקה. הדיוק המאוזן הממוצע בבדיקה זו הוא **81.97%**, עם סטיית תקן של **8.14 נקודות אחוז** בין החלוקות. פירוט הבדיקה נמצא ב־[outputs/final_pipeline_audit.json](outputs/final_pipeline_audit.json).

הנתונים כוללים 25 הקלטות משמונה מצבים ניסויים בגשר אחד. בחירת המאפיינים והפרמטרים נעשתה לאחר ניסויים, ואין קבוצת בדיקה סופית שלא שימשה במהלך המחקר. המודל הארוז אומן לבסוף על כל 25 ההקלטות; חיזוי שלהן אינו מדד לביצועים על נתונים חדשים.

ראו את [כרטיס המודל](prediction/artifacts/model_card.md) למגבלות המלאות. מודל ה־Holdout הנפרד נשמר ב־`models/bridge_damage_rf_group_holdout.joblib`; הוא אינו המודל שהחיזוי טוען כברירת מחדל. תוצאות של ניסויים מוקדמים נשמרו לצורך תיעוד, ויש להבחין ביניהן לבין בדיקת המודל הסופי.

## סביבת המחקר והמחברת

לאחר יצירת הסביבה והוספת קובץ הנתונים, התקינו את תלויות המחקר ורשמו את ה־kernel של המחברת:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name bridgepulse --display-name "Python (BridgePulse)"
.\.venv\Scripts\python.exe -m jupyterlab notebooks/experiment.ipynb
```

בחרו במחברת את ה־kernel בשם **Python (BridgePulse)**. המחברת כוללת ניתוח אותות, השוואת מודלים ותוצאות ניסויים. הרצת כל התאים מבצעת חישובים ואימון מחדש, ועשויה לעדכן קבצים ב־`data/processed/`, ב־`models/` וב־`outputs/`.

`requirements-lock.txt` מתעד את גרסאות סביבת המחקר; `requirements.txt` מפרט את תלויות המחקר הישירות. תלויות החיזוי בלבד נמצאות ב־`prediction/requirements.txt`.

## מבנה הפרויקט

| נתיב | תוכן |
|---|---|
| `prediction/` | חבילת חיזוי עצמאית, חילוץ מאפיינים ומודל מוכן ב־`artifacts/model.joblib` |
| `data/raw/` | חתימת קובץ המקור; את קובץ ה־H5 עצמו יש להוסיף מקומית |
| `data/processed/` | מאפיינים מחושבים שנשמרו לצורך ניסויים |
| `notebooks/experiment.ipynb` | מחברת המחקר והתוצאות |
| `scripts/` | קוד ניתוח, בדיקות, השוואת מודלים ורישום ל־MLflow |
| `models/` | מודלי המחקר וכרטיס המודל |
| `outputs/` | דוחות JSON, גרפים ודוגמת חיזוי ב־`predictions/` |

הסביבה `.venv/`, גיבויי `archive/`, קובצי לוג ומאגרי MLflow מקומיים אינם נכללים ב־Git.

## MLflow: רישום מקומי אופציונלי

המודל תועד בניסוי המקורי בשם `BridgePulseDamageRF`, גרסה 1. פרטי הרישום והמדדים נמצאים ב־[outputs/mlflow_registration.json](outputs/mlflow_registration.json); בדיקות ההתאמה של חבילת החיזוי נמצאות ב־[outputs/prediction_package_checks.json](outputs/prediction_package_checks.json).

מסד הנתונים והארטיפקטים של MLflow מהסביבה המקורית אינם נכללים במאגר. כדי ליצור רישום בסביבה שלכם, לאחר התקנת תלויות המחקר והוספת קובץ המקור:

```powershell
.\.venv\Scripts\python.exe scripts/register_bridge_model.py
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

סקריפט הרישום מריץ מחדש את בדיקות המודל, מאמן אותו ומעדכן ארטיפקטים ודוחות. כברירת מחדל הוא יוצר מאגר MLflow מקומי; אם הוגדר `MLFLOW_TRACKING_URI`, הוא משתמש בו. פקודת הממשק המוצגת פותחת את המאגר המקומי ב־http://127.0.0.1:5000. רישום המודל אינו פריסה שלו לשירות.
