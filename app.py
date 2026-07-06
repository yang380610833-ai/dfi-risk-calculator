"""
DFI 院内死亡风险预测 — 在线计算器
启动方式：python app.py
访问地址：http://127.0.0.1:5000
"""
import json
import os
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# 加载模型参数
with open("model_params.json") as f:
    mp = json.load(f)
with open("input_transform.json") as f:
    it = json.load(f)

FEATURES      = mp["features"]
INTERCEPT     = mp["intercept"]
COEFFICIENTS  = np.array(mp["coefficients"])
TRAIN_MEAN    = np.array(it["train_mean"])
TRAIN_STD     = np.array(it["train_std"])
THRESHOLDS    = mp["thresholds"]         # [0.034, 0.103, 0.298]
THRESH_LABELS = mp["threshold_labels"]   # ["低风险","中风险","高风险","极高风险"]

# 变量中文名（前端展示用）
FEATURE_LABELS = {
    "anchor_age":       "年龄（岁）",
    "mv_24h":           "24h内机械通气",
    "debridement":      "清创术",
    "charlson_score":   "Charlson合并症指数",
    "hr_mean":          "平均心率（bpm）",
    "lactate_baseline": "基线乳酸（mmol/L）",
    "sofa_baseline":    "基线SOFA评分",
    "rdw_first":        "首次RDW（%）",
    "map_mean":         "平均MAP（mmHg）",
    "inr_first":        "首次INR",
}


def stratify_risk(prob):
    """根据阈值返回风险分层"""
    if prob < THRESHOLDS[0]:
        return THRESH_LABELS[0]   # 低风险
    elif prob < THRESHOLDS[1]:
        return THRESH_LABELS[1]   # 中风险
    elif prob < THRESHOLDS[2]:
        return THRESH_LABELS[2]   # 高风险
    else:
        return THRESH_LABELS[3]   # 极高风险


@app.route("/")
def index():
    return render_template("index.html",
                           features=FEATURES,
                           labels=FEATURE_LABELS,
                           thresholds=THRESHOLDS,
                           thresh_labels=THRESH_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    """接收 JSON 格式的输入，返回概率和风险分层"""
    data = request.get_json()
    x = np.array([float(data[feat]) for feat in FEATURES])

    # z-score 标准化（使用训练集的 μ 和 σ）
    x_norm = (x - TRAIN_MEAN) / TRAIN_STD

    # 线性预测值 LP
    lp = INTERCEPT + np.dot(COEFFICIENTS, x_norm)

    # LP → 概率
    prob = 1.0 / (1.0 + np.exp(-lp))
    risk = stratify_risk(prob)

    return jsonify({
        "probability": round(float(prob), 4),
        "risk_level":  risk,
        "lp":          round(float(lp), 4),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
