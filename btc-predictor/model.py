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

    # Target : 1 if price goes up tomorrow (NaN for the last row — no next day)
    data['y'] = (data['Close'].shift(-1) > data['Close']).astype(float)
    data.loc[data.index[-1], 'y'] = np.nan  # last row has no known outcome yet

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

    # Keep original df_feat index so we can look up next row correctly
    df_known = df_feat.dropna(subset=FEATURES + ['y'])  # NO reset_index
    hist_slice = df_known.tail(n_history)

    X_hist = scaler.transform(hist_slice[FEATURES].values)
    pred_hist = model.predict(X_hist)

    history = []
    for i, (orig_idx, row) in enumerate(hist_slice.iterrows()):
        # orig_idx is the RangeIndex of df_feat → orig_idx+1 is the actual next day
        if orig_idx + 1 in df_feat.index:
            next_close = float(df_feat.loc[orig_idx + 1, 'Close'])
        else:
            next_close = None

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
            'next_close':   round(next_close, 2) if next_close is not None else None,
            'daily_return': round(daily_return * 100, 3) if daily_return is not None else None,
            'actual_up':    int(row['y']),
            'predicted_up': predicted,
            'correct':      int(predicted == int(row['y'])),
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
