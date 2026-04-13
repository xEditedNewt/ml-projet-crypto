"""
KNN BTC predictor — same logic as projet_ml_crypto.ipynb
"""
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

FEATURES = [
    'return_1', 'return_2', 'return_5', 'return_10',
    'ecart_MA20',
    'vol_5', 'vol_20',
    'vol_norm',
]


def fetch_btc(start='2018-01-01'):
    raw = yf.download('BTC-USD', start=start, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={'Date': 'Date'})
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def build_features(df):
    data = df.copy()
    data['return_1']  = data['Close'].pct_change(1)
    data['return_2']  = data['Close'].pct_change(2)
    data['return_5']  = data['Close'].pct_change(5)
    data['return_10'] = data['Close'].pct_change(10)

    data['MA20']      = data['Close'].rolling(20).mean()
    data['ecart_MA20'] = (data['Close'] - data['MA20']) / data['MA20']

    data['vol_5']  = data['return_1'].rolling(5).std()
    data['vol_20'] = data['return_1'].rolling(20).std()

    data['vol_norm'] = data['Volume'] / data['Volume'].rolling(20).mean()

    # Target : 1 if price goes up tomorrow
    data['y'] = (data['Close'].shift(-1) > data['Close']).astype(float)

    return data


def train_knn(df_feat):
    df_clean = df_feat.dropna(subset=FEATURES + ['y']).reset_index(drop=True)

    X = df_clean[FEATURES].values
    y = df_clean['y'].values

    # 95 / 5 split
    split_idx = int(len(df_clean) * 0.95)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, _ = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {'n_neighbors': [5, 10, 15, 20, 30, 50]}
    knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=tscv, scoring='accuracy')
    knn_cv.fit(X_train_sc, y_train)

    return knn_cv.best_estimator_, scaler, df_clean


def predict_today_and_history(df_feat, model, scaler, n_history=5):
    """
    Returns:
      - today_signal: 1 (up) or 0 (down)  — prediction for today's close vs tomorrow
      - history: list of dicts {date, actual_up, predicted_up, correct}
      - chart_data: last 30 days {dates, prices}
      - current_price: float
    """
    df_all = df_feat.dropna(subset=FEATURES).reset_index(drop=True)

    # Current price = last close
    current_price = float(df_all['Close'].iloc[-1])

    # Chart: last 30 rows
    chart_slice = df_all.tail(30)
    chart_dates  = chart_slice['Date'].dt.strftime('%Y-%m-%d').tolist()
    chart_prices = chart_slice['Close'].round(2).tolist()

    # Rows that have both features AND known y (all but the last)
    df_known = df_feat.dropna(subset=FEATURES + ['y']).reset_index(drop=True)
    # Last n_history rows with known outcome
    hist_slice = df_known.tail(n_history)

    X_hist = scaler.transform(hist_slice[FEATURES].values)
    pred_hist = model.predict(X_hist)

    # Next-day close needed to compute actual return
    # df_known row i has Close[i], and y[i] = 1 iff Close[i+1] > Close[i]
    # We need Close[i+1] — grab it from df_feat aligned by index
    hist_indices = hist_slice.index.tolist()

    history = []
    for i, (idx, row) in enumerate(hist_slice.iterrows()):
        # Find next row in df_feat to get actual next close
        next_rows = df_feat[df_feat.index == idx + 1]
        if len(next_rows) == 0:
            # fallback: look by position in df_known
            pos = df_known.index.get_loc(idx)
            if pos + 1 < len(df_known):
                next_close = float(df_known.iloc[pos + 1]['Close'])
            else:
                next_close = None
        else:
            next_close = float(next_rows.iloc[0]['Close'])

        close = float(row['Close'])
        # Daily return: (next_close - close) / close
        if next_close is not None:
            daily_return = (next_close - close) / close  # e.g. 0.025 = +2.5%
        else:
            daily_return = None

        # Gain on 100€ bet:
        # LONG  (predicted up)   → gain = 100 * daily_return
        # SHORT (predicted down) → gain = 100 * (-daily_return)
        predicted = int(pred_hist[i])
        if daily_return is not None:
            direction = 1 if predicted == 1 else -1
            gain_eur = round(100 * direction * daily_return, 2)
        else:
            gain_eur = None

        history.append({
            'date':         row['Date'].strftime('%d/%m/%Y'),
            'close':        round(close, 2),
            'next_close':   round(next_close, 2) if next_close else None,
            'daily_return': round(daily_return * 100, 3) if daily_return is not None else None,
            'actual_up':    int(row['y']),
            'predicted_up': predicted,
            'correct':      int(predicted == row['y']),
            'gain_eur':     gain_eur,
            'position':     'LONG' if predicted == 1 else 'SHORT',
        })

    # Today's prediction (last row in df_all — no known y yet)
    last_row = df_all.tail(1)
    if last_row[FEATURES].isna().any(axis=1).iloc[0]:
        today_signal = None
    else:
        X_today = scaler.transform(last_row[FEATURES].values)
        today_signal = int(model.predict(X_today)[0])

    return {
        'today_signal':  today_signal,
        'history':       history,
        'chart_dates':   chart_dates,
        'chart_prices':  chart_prices,
        'current_price': current_price,
        'today_date':    df_all['Date'].iloc[-1].strftime('%d/%m/%Y'),
    }
