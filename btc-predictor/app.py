"""
Flask app — BTC predictor (KNN / Logistic / Ridge / Lasso)
Model cache: refreshed at most once per hour, per model type.
"""
import time
import threading
from flask import Flask, jsonify, render_template, request
from model import (
    fetch_btc, build_features, train_model, predict_today_and_history,
    fetch_live_price, VALID_MODELS,
)

app = Flask(__name__)

# ---------- Model result cache (keyed by model name) ----------
_cache: dict = {}          # { model_name: {'data': ..., 'timestamp': float} }
_lock  = threading.Lock()
CACHE_TTL = 3600           # 1 hour

# ---------- Live price cache ----------
_price_cache = {'price': None, 'timestamp': 0.0}
_price_lock  = threading.Lock()
PRICE_TTL = 60             # 1 minute


def get_prediction(model_name: str = 'knn'):
    now = time.time()
    with _lock:
        entry = _cache.setdefault(model_name, {'data': None, 'timestamp': 0.0})
        if entry['data'] is None or (now - entry['timestamp']) > CACHE_TTL:
            df_raw      = fetch_btc()
            df_feat     = build_features(df_raw)
            model, scaler, test_acc = train_model(df_feat, model_name)
            result      = predict_today_and_history(df_feat, model, scaler)
            result['model_name']    = model_name
            result['test_accuracy'] = round(test_acc * 100, 1)
            entry['data']      = result
            entry['timestamp'] = now
    return entry['data']


def get_live():
    now = time.time()
    with _price_lock:
        if _price_cache['price'] is None or (now - _price_cache['timestamp']) > PRICE_TTL:
            _price_cache['price']     = fetch_live_price()
            _price_cache['timestamp'] = now
        return _price_cache['price']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/prediction')
def prediction():
    try:
        model_name = request.args.get('model', 'knn').lower()
        if model_name not in VALID_MODELS:
            model_name = 'knn'
        data = get_prediction(model_name)
        return jsonify({'status': 'ok', **data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/price')
def live_price():
    try:
        return jsonify({'status': 'ok', 'price': get_live()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=8080, use_reloader=False)
