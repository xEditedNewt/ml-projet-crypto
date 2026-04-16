"""
KNN BTC predictor — same logic as projet_ml_crypto.ipynb
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

# Local CSV from Kaggle (Binance BTC/USDT) — same data source as the notebook
_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'btc_1d_data_2018_to_2025.csv')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score

FEATURES = [
    'return_1', 'return_2', 'return_5', 'return_10',
    'ecart_MA20',
    'vol_5', 'vol_20',
    'vol_norm',
]


def fetch_live_price():
    """Near-realtime BTC price via yfinance fast_info (crypto = 24/7)."""
    ticker = yf.Ticker('BTC-USD')
    return float(ticker.fast_info.last_price)


def _strip_tz(series):
    """Return a tz-naive datetime Series regardless of input timezone."""
    s = pd.to_datetime(series)
    if s.dt.tz is not None:
        return s.dt.tz_convert(None)
    return s


def fetch_btc(start='2018-01-01'):
    """
    Load from the local Kaggle CSV (identical source to the notebook) then
    extend with yfinance only for days after the CSV ends.
    This ensures training data — and therefore model accuracy — match the notebook.
    """
    frames = []
    csv_last_date = None

    if os.path.exists(_CSV_PATH):
        df_csv = pd.read_csv(_CSV_PATH)
        df_csv['Date'] = _strip_tz(df_csv['Open time'])
        df_csv = df_csv[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_csv = df_csv.sort_values('Date').reset_index(drop=True)
        frames.append(df_csv)
        csv_last_date = df_csv['Date'].max()

    # Extend with yfinance for days the CSV doesn't cover yet
    yf_start = (csv_last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if csv_last_date is not None else start
    raw = yf.download('BTC-USD', start=yf_start, progress=False, auto_adjust=False)
    if not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df_yf = raw.reset_index()
        df_yf['Date'] = _strip_tz(df_yf['Date'])
        df_yf = df_yf[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        frames.append(df_yf)

    df = (pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset='Date')
          .sort_values('Date')
          .reset_index(drop=True))
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


VALID_MODELS = ('knn', 'logistic', 'ridge', 'lasso')

MODEL_LABELS = {
    'knn':      'KNN',
    'logistic': 'Régression Logistique',
    'ridge':    'Ridge (L₂)',
    'lasso':    'Lasso (L₁)',
}


def train_model(df_feat, model_name='knn'):
    """Train any of the four models from the notebook — same split/scaler logic."""
    df_clean = df_feat.dropna(subset=FEATURES + ['y']).reset_index(drop=True)

    X = df_clean[FEATURES].values
    y = df_clean['y'].values

    split_idx = int(len(df_clean) * 0.95)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    tscv = TimeSeriesSplit(n_splits=5)

    if model_name == 'knn':
        param_grid = {'n_neighbors': [3, 5, 10, 15, 20, 30, 50, 75, 100]}
        clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=tscv, scoring='accuracy')
    elif model_name == 'logistic':
        clf = LogisticRegression(penalty=None, max_iter=2000)
    elif model_name == 'ridge':
        clf = LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 20), cv=tscv, penalty='l2',
            solver='lbfgs', scoring='accuracy', max_iter=2000,
        )
    elif model_name == 'lasso':
        clf = LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 20), cv=tscv, penalty='l1',
            solver='liblinear', scoring='accuracy', max_iter=2000,
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')

    clf.fit(X_train_sc, y_train)
    best_clf = clf.best_estimator_ if hasattr(clf, 'best_estimator_') else clf

    test_acc = float(accuracy_score(y_test, best_clf.predict(X_test_sc)))
    return best_clf, scaler, test_acc


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


def predict_today_and_history(df_feat, model, scaler, n_history=60):
    """
    Returns:
      - today_signal: 1 (up) or 0 (down) — based on yesterday's close only (stable)
      - signal_basis_date: the closed trading day the signal is computed from
      - history: list of dicts {date, actual_up, predicted_up, correct}
      - chart_data: last 30 days {dates, prices}
      - current_price: float (last known close)
      - yesterday_close: float
      - price_change_pct: float
    """
    df_all = df_feat.dropna(subset=FEATURES).reset_index(drop=True)

    # Prices
    current_price  = float(df_all['Close'].iloc[-1])
    yesterday_close = float(df_all['Close'].iloc[-2]) if len(df_all) >= 2 else None
    price_change_pct = (
        round((current_price - yesterday_close) / yesterday_close * 100, 2)
        if yesterday_close else None
    )

    # Chart: last 30 rows
    chart_slice  = df_all.tail(30)
    chart_dates  = chart_slice['Date'].dt.strftime('%Y-%m-%d').tolist()
    chart_prices = chart_slice['Close'].round(2).tolist()

    # Keep original df_feat index so we can look up next row correctly
    df_known   = df_feat.dropna(subset=FEATURES + ['y'])  # NO reset_index
    hist_slice = df_known.tail(n_history)

    X_hist      = scaler.transform(hist_slice[FEATURES].values)
    pred_hist   = model.predict(X_hist)
    proba_hist  = model.predict_proba(X_hist)   # shape (n, 2) — columns: [P(down), P(up)]

    history = []
    for i, (orig_idx, row) in enumerate(hist_slice.iterrows()):
        if orig_idx + 1 in df_feat.index:
            next_close = float(df_feat.loc[orig_idx + 1, 'Close'])
            next_high  = float(df_feat.loc[orig_idx + 1, 'High'])
            next_low   = float(df_feat.loc[orig_idx + 1, 'Low'])
        else:
            next_close = None
            next_high  = None
            next_low   = None

        close = float(row['Close'])
        daily_return = (next_close - close) / close if next_close is not None else None

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
            'next_high':    round(next_high, 2)  if next_high  is not None else None,
            'next_low':     round(next_low, 2)   if next_low   is not None else None,
            'daily_return': round(daily_return * 100, 3) if daily_return is not None else None,
            'actual_up':    int(row['y']),
            'predicted_up': predicted,
            'correct':      int(predicted == int(row['y'])),
            'gain_eur':     gain_eur,
            'position':     'LONG' if predicted == 1 else 'SHORT',
            'proba':        round(float(proba_hist[i][predicted]) * 100, 3),
        })

    # ── Today's signal ──────────────────────────────────────────────────────
    # Uses YESTERDAY's features (iloc[-2]) so today's intraday price has zero
    # influence on the signal — the prediction is fully determined at yesterday's close.
    today_proba = None
    if len(df_all) < 2:
        today_signal      = None
        signal_basis_date = None
    else:
        signal_row = df_all.iloc[[-2]]
        if signal_row[FEATURES].isna().any(axis=1).iloc[0]:
            today_signal      = None
            signal_basis_date = None
        else:
            X_signal          = scaler.transform(signal_row[FEATURES].values)
            today_signal      = int(model.predict(X_signal)[0])
            today_proba       = round(float(model.predict_proba(X_signal)[0][today_signal]) * 100, 3)
            signal_basis_date = df_all['Date'].iloc[-2].strftime('%d/%m/%Y')

    return {
        'today_signal':      today_signal,
        'today_proba':       today_proba if today_signal is not None else None,
        'signal_basis_date': signal_basis_date,
        'history':           history,
        'chart_dates':       chart_dates,
        'chart_prices':      chart_prices,
        'current_price':     current_price,
        'yesterday_close':   yesterday_close,
        'price_change_pct':  price_change_pct,
        'today_date':        df_all['Date'].iloc[-1].strftime('%d/%m/%Y'),
    }
