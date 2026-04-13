# BTC Predictor

Site web illustrant le projet ML — prédiction KNN hausse/baisse du Bitcoin.

## Lancement local

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

## Hébergement

Compatible **Render**, **Railway**, ou tout hébergeur Python.  
Ajoute un `Procfile` si besoin :
```
web: gunicorn app:app
```
Et installe gunicorn : `pip install gunicorn`.

## Fonctionnement

- `/` — page principale
- `/api/prediction` — JSON avec signal du jour, historique 5 jours, courbe 30j
- Cache en mémoire : les données sont rafraîchies **au maximum 1 fois par heure**
- À chaque rafraîchissement : téléchargement BTC-USD (2018→aujourd'hui) via yfinance, entraînement KNN (GridSearch k ∈ {5,10,15,20,30,50}), prédiction
