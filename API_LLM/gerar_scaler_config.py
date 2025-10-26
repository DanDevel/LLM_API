# gerar_scaler_config.py
import torch, json, os

MODEL_PATH = "outputs/model/bitcoinhomebroker_latest.pth"
MODEL_DIR  = os.path.dirname(MODEL_PATH)

bundle = torch.load(MODEL_PATH, map_location="cpu")

# tenta extrair
scaler = bundle.get("scaler", None)
config = bundle.get("config", None)
hparams = bundle.get("hparams", {})

# cria paths
scaler_path = os.path.join(MODEL_DIR, "scaler.json")
config_path = os.path.join(MODEL_DIR, "config.json")

if scaler:
    with open(scaler_path, "w") as f:
        json.dump(scaler, f)
    print(f"scaler.json criado: {scaler_path}")
else:
    print("⚠️ bundle não contém scaler, será necessário retreinar para gerar esse arquivo.")

if config or hparams:
    cfg = {}
    if config: cfg.update(config)
    if hparams: cfg.update(hparams)
    with open(config_path, "w") as f:
        json.dump(cfg, f)
    print(f"config.json criado: {config_path}")
else:
    print("⚠️ bundle não contém config; retreine para gerar corretamente.")
