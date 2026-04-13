"""
Flask app — BTC KNN predictor
Cache: data refreshed at most once per hour
"""
import time
import threading
from flask import Flask, jsonify, render_template
from model import fetch_btc, build_features, train_knn, predict_today_and_history

app = Flask(__name__)

# ---------- In-memory cache ----------
_cache = {
    'data': None,
    'timestamp': 0,
}
_lock = threading.Lock()
CACHE_TTL = 3600  # 1 hour


def get_prediction():
    now = time.time()
    with _lock:
        if _cache['data'] is None or (now - _cache['timestamp']) > CACHE_TTL:
            df_raw  = fetch_btc()
            df_feat = build_features(df_raw)
            model, scaler, _ = train_knn(df_feat)
            result  = predict_today_and_history(df_feat, model, scaler)
            _cache['data']      = result
            _cache['timestamp'] = now
        return _cache['data']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/prediction')
def prediction():
    try:
        data = get_prediction()
        return jsonify({'status': 'ok', **data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=8080, use_reloader=False)
