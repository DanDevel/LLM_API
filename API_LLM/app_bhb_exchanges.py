# app_bhb_exchanges.py
# BitcoinHomeBroker – Flask API + UI com seletor de exchange (Binance/KuCoin/Poloniex)
# Compatível com bundle .pth (pesos + hparams + scaler + config) e fallback JSON.

import os, json, time
from typing import List, Tuple, Dict
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template_string

import torch
import torch.nn as nn

# ---------------------------
# Caminhos de artefatos
# ---------------------------
MODEL_DIR   = os.environ.get("MODEL_DIR", "outputs/model")
MODEL_PATH  = os.path.join(MODEL_DIR, "bitcoinhomebroker_latest.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# ---------------------------
# Arquitetura do modelo (igual ao treino)
# ---------------------------
class LSTM_PT(nn.Module):
    def __init__(self, n_features=1, hidden=128, layers_n=2, dropout=0.2, bidir=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers_n,
            batch_first=True,
            dropout=(dropout if layers_n > 1 else 0.0),
            bidirectional=bidir
        )
        out_dim = hidden * (2 if bidir else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

# ---------------------------
# Carregamento do bundle + fallback
# ---------------------------
def _load_bundle(path):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        payload = {"state_dict": payload}
    return payload

def _load_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

BUNDLE     = _load_bundle(MODEL_PATH)
STATE_DICT = BUNDLE.get("state_dict")
HPARAMS    = BUNDLE.get("hparams", {}) or {}
SCALER     = BUNDLE.get("scaler") or _load_json(SCALER_PATH, default=None)
CONFIG     = BUNDLE.get("config") or _load_json(CONFIG_PATH, default={})

LOOKBACK = int(CONFIG.get("lookback", 48))
HORIZON  = int(CONFIG.get("horizon", 1))

cfg_arch = {
    "hidden":   int(HPARAMS.get("hidden", 192)),
    "layers_n": int(HPARAMS.get("layers", 2)),
    "dropout":  float(HPARAMS.get("dropout", 0.2)),
    "bidir":    bool(HPARAMS.get("bidir", False)),
}
model = LSTM_PT(n_features=1, **cfg_arch)
model.load_state_dict(STATE_DICT)
model.eval()

if SCALER is None:
    raise RuntimeError("Scaler não encontrado (bundle .pth sem 'scaler' e scaler.json ausente).")

# ---------------------------
# Utils
# ---------------------------
def standardize_close(arr: np.ndarray, scaler: dict) -> Tuple[np.ndarray, float, float]:
    mean  = float(scaler["mean"]); scale = float(scaler["scale"])
    return (arr - mean) / (scale + 1e-12), mean, scale

def inverse_standardize(x: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return x * (scale + 1e-12) + mean

# --- Fetchers 15m por exchange ---
def fetch_binance_closes_15m(symbol="BTCUSDT", limit=500) -> List[float]:
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": "15m", "limit": limit}, timeout=20)
    r.raise_for_status()
    return [float(k[4]) for k in r.json()]  # close no índice 4

def fetch_kucoin_closes_15m(symbol="BTC-USDT", limit=500) -> List[float]:
    # KuCoin exige startAt/endAt (segundos). Calculamos janela mínima.
    now_s = int(time.time())
    start_s = now_s - limit * 15 * 60
    url = "https://api.kucoin.com/api/v1/market/candles"
    params = {"symbol": symbol, "type": "15min", "startAt": start_s, "endAt": now_s}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if payload.get("code") != "200000":
        raise RuntimeError(f"KuCoin erro: {payload}")
    data = payload.get("data", [])  # mais recente primeiro
    data = list(reversed(data))      # invertendo para cronológico
    closes = [float(x[2]) for x in data]  # [time, open, close, high, low, volume, turnover]
    return closes[-limit:]

def fetch_poloniex_closes_15m(market="BTC_USDT", limit=500) -> List[float]:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - limit * 15 * 60 * 1000
    url = f"https://api.poloniex.com/markets/{market}/candles"
    params = {"startTime": start_ms, "endTime": now_ms, "interval": "MINUTE_15"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    arr = r.json()  # conforme exemplo: close no índice 3
    closes = [float(x[3]) for x in arr]
    return closes[-limit:]

EXCHANGE_FETCH: Dict[str, Dict[str, str]] = {
    "binance":   {"fn": "binance",   "symbol": "BTCUSDT"},
    "kucoin":    {"fn": "kucoin",    "symbol": "BTC-USDT"},
    "poloniex":  {"fn": "poloniex",  "symbol": "BTC_USDT"},
}

def fetch_closes(exchange: str, limit_needed: int) -> List[float]:
    ex = exchange.lower()
    if ex not in EXCHANGE_FETCH:
        raise ValueError("exchange inválida. Use binance, kucoin ou poloniex.")
    meta = EXCHANGE_FETCH[ex]
    if meta["fn"] == "binance":
        return fetch_binance_closes_15m(meta["symbol"], limit=limit_needed)
    if meta["fn"] == "kucoin":
        return fetch_kucoin_closes_15m(meta["symbol"], limit=limit_needed)
    return fetch_poloniex_closes_15m(meta["symbol"], limit=limit_needed)

def steps_from_minutes(mins: int) -> int:
    return max(1, int(round(mins / 15.0)))  # cada passo = 15 min

def predict_iterative(closes_15m: List[float], steps: int) -> dict:
    if len(closes_15m) < LOOKBACK:
        raise ValueError(f"Precisa de pelo menos {LOOKBACK} closes; recebi {len(closes_15m)}.")
    window = np.array(closes_15m[-LOOKBACK:], dtype=np.float32)
    win_std, mean, scale = standardize_close(window, SCALER)
    cur = win_std.copy()
    preds_std = []
    for _ in range(steps):
        x = torch.tensor(cur.reshape(1, LOOKBACK, 1), dtype=torch.float32)
        with torch.no_grad():
            yhat_std = model(x).cpu().numpy().reshape(-1)[0]
        preds_std.append(yhat_std)
        cur = np.concatenate([cur[1:], [yhat_std]], axis=0)
    preds_usd = inverse_standardize(np.array(preds_std, dtype=np.float32), mean, scale).tolist()
    return {
        "last_close_usd": float(window[-1]),
        "pred_path_usd": [float(v) for v in preds_usd],
        "final_pred_usd": float(preds_usd[-1]),
    }

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <title>BitcoinHomeBroker • Previsão BTC/USDT</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { color-scheme: light dark; }
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif; }
    header { padding: 16px 20px; background: linear-gradient(90deg,#0f172a,#334155); color:#fff; }
    main { max-width: 900px; margin: 24px auto; padding: 0 16px; }
    .card { background: rgba(255,255,255,0.06); border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 16px; }
    label { display:block; margin-bottom: 6px; font-weight: 600; }
    select, button { padding: 10px 12px; border-radius: 10px; border: 1px solid #94a3b8; font-size: 15px; }
    button { cursor: pointer; background: #0ea5e9; color: white; border: none; }
    button:hover { filter: brightness(0.95); }
    .row { display: grid; grid-template-columns: 1fr 1fr auto; gap: 12px; align-items: end; }
    .muted { opacity: .8; font-size: 13px; }
    .result { margin-top: 16px; padding: 12px; border-radius: 10px; background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.35); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    footer { padding: 20px; text-align:center; color:#64748b; }
    .error { margin-top: 16px; padding: 12px; border-radius: 10px; background: rgba(239,68,68,0.10); border: 1px solid rgba(239,68,68,0.35); }
  </style>
</head>
<body>
  <header>
    <h1>Previsão BTC/USDT</h1>
    <div class="muted">Janela base: {{ lookback }} candles de 15 min • Modelo LSTM (PyTorch)</div>
  </header>

  <main>
    <div class="card">
      <form method="post" action="/ask">
        <label for="exchange">Exchange</label>
        <div class="row">
          <select id="exchange" name="exchange" required>
            <option value="binance">Binance (BTCUSDT)</option>
            <option value="kucoin">KuCoin (BTC-USDT)</option>
            <option value="poloniex">Poloniex (BTC_USDT)</option>
          </select>

          <select id="minutes" name="minutes_ahead" required>
            <option value="15">Próximos 15 minutos</option>
            <option value="60">Próxima 1 hora</option>
          </select>

          <button type="submit">Enviar</button>
        </div>
        <p class="muted">As previsões são iterativas em blocos de 15 minutos.</p>
      </form>

      {% if error %}
        <div class="error"><b>Erro:</b> {{ error }}</div>
      {% endif %}

      {% if result %}
      <div class="result">
        <h3>Resultado</h3>
        <p>Exchange: <b>{{ result.exchange }}</b></p>
        <p>Último close (USD): <b class="mono">{{ result.last_close_usd }}</b></p>
        <p>Passos previstos (15m): <span class="mono">{{ result.pred_path_usd }}</span></p>
        <p>Valor estimado em {{ result.label }}: <b class="mono">{{ result.final_pred_usd }}</b></p>
      </div>
      {% endif %}
    </div>
  </main>

  <footer>© {{ year }} • BitcoinHomeBroker</footer>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML, lookback=LOOKBACK, year=time.gmtime().tm_year, result=None, error=None)

@app.post("/ask")
def ask_html():
    ex   = (request.form.get("exchange") or "binance").lower()
    mins = int(request.form.get("minutes_ahead", 15))
    steps = steps_from_minutes(mins)
    try:
        closes = fetch_closes(ex, limit_needed=max(LOOKBACK + steps, 200))
        out = predict_iterative(closes, steps)
        label = f"{mins} min" if mins < 60 else "1 hora"
        result = {
            "exchange": ex,
            "last_close_usd": round(out["last_close_usd"], 2),
            "pred_path_usd": [round(x, 2) for x in out["pred_path_usd"]],
            "final_pred_usd": round(out["final_pred_usd"], 2),
            "label": label
        }
        return render_template_string(HTML, lookback=LOOKBACK, year=time.gmtime().tm_year, result=result, error=None)
    except Exception as e:
        return render_template_string(HTML, lookback=LOOKBACK, year=time.gmtime().tm_year, result=None, error=str(e))

# -------- API JSON --------
@app.post("/api/ask")
def ask_api():
    """
    Body:
    {
      "minutes_ahead": 15,          # 15 ou 60 (ou múltiplos de 15)
      "exchange": "binance"         # binance | kucoin | poloniex
    }
    """
    data = request.get_json(silent=True) or {}
    mins = int(data.get("minutes_ahead", 15))
    ex   = (data.get("exchange") or "binance").lower()
    steps = steps_from_minutes(mins)
    try:
        closes = fetch_closes(ex, limit_needed=max(LOOKBACK + steps, 200))
        out = predict_iterative(closes, steps)
        return jsonify({
            "ok": True,
            "exchange": ex,
            "lookback": LOOKBACK,
            "step_minutes": 15,
            "minutes_ahead": mins,
            "last_close_usd": out["last_close_usd"],
            "pred_path_usd": out["pred_path_usd"],
            "final_pred_usd": out["final_pred_usd"]
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "lookback": LOOKBACK,
        "horizon_base": HORIZON,
        "has_scaler": SCALER is not None,
        "model_hparams": cfg_arch,
        "exchanges": list(EXCHANGE_FETCH.keys())
    })

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
