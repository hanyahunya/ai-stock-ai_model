from .volumeModel import train_ai_model
from tensorflow.keras.models import load_model
import tensorflow as tf
from typing import Optional, Tuple
import joblib
from pathlib import Path
from . import normalization as norm
import numpy as np
# import os
# import json
# from dotenv import load_dotenv
# load_dotenv()
from pprint import pprint


# def peek_pair(name_x, name_y, X, y, n=5):
#     """
#     X[0] ↔ y[0], X[1] ↔ y[1] … 식으로 앞 n개를 짝지어 출력.
#     X : 2D 또는 1D (list‧ndarray)
#     y : 1D (list‧ndarray)
#     """
#     X_arr = np.asarray(X, dtype=object)
#     y_arr = np.asarray(y, dtype=object)
#     n = min(n, len(X_arr), len(y_arr))

#     print(f"\n┌── {name_x} × {name_y} ── (first {n}) ───────────────")
#     for i in range(n):
#         print(f"│ [{i:>4}]  X = {X_arr[i]}   |   y = {y_arr[i]}")
#     print("└──────────────────────────────────────────────\n")


def make_lstm_dataset(
    X_raw: np.ndarray,
    y_raw: Optional[np.ndarray] = None,
    sequence_length: int = 30 # --------------------------시퀀스------------------
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X_seq, y_seq = [], []

    for i in range(sequence_length - 1, len(X_raw)):
        X_seq.append(X_raw[i - sequence_length + 1 : i + 1])
        if y_raw is not None:
            y_seq.append(y_raw[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq) if y_raw is not None else None
    return X_seq, y_seq

def train_model(x_data, y_data, stockCode):
    # X 데이터 분리
    x_volume, x_investor, x_short = [], [], []
    for row in x_data:
        base = list(row[0:6]) # 시고저종 등락률 거래량 0 1 2 3 4 5

        # x_volume.append(base + [row[5]]) # ohlc + 거래량
        x_volume.append(base + list(row[15:]) + list(row[6:15])) # 시고저종, 거래량, 

        x_investor.append(base + list(row[6:10]))
        x_short.append(base + list(row[10:15]))

    # OHLC(0~3)만 공통 스케일러로 묶음 ↓↓↓
    x_volume_np, (x_volume_group_min, x_volume_group_max), x_volume_other_scalers = norm.normalize_2d_array(x_volume, shared_idx=[0, 1, 2, 3, 6, 7, 8, 9, 10])

    x_investor_np, (x_investor_group_min, x_investor_group_max), x_investor_other_scalers = \
        norm.normalize_2d_array(x_investor, shared_idx=[0, 1, 2, 3])

    x_short_np, (x_short_group_min, x_short_group_max), x_short_other_scalers = \
        norm.normalize_2d_array(x_short, shared_idx=[0, 1, 2, 3])

    save_scalers_x(x_volume_group_min, x_volume_group_max, x_volume_other_scalers, "volume", stockCode)
    save_scalers_x(x_investor_group_min, x_investor_group_max, x_investor_other_scalers, "investor", stockCode)
    save_scalers_x(x_short_group_min, x_short_group_max, x_short_other_scalers, "short", stockCode)


    # inference(x_volume, stockCode, x_volume_np)

    # Y 데이터 분리
    y_highest_ratio = [row[0] for row in y_data]
    y_lowest_ratio = [row[1] for row in y_data]
    y_is_up = [row[2] for row in y_data]
    y_is_down = [row[3] for row in y_data]

    # peek_pair("x_volume", "y_is_up", x_volume, y_is_up, n=5)

    y_highest_np, y_highest_scalers = norm.normalize_1d_array(y_highest_ratio)

    y_lowest_np, y_lowest_scalers = norm.normalize_1d_array(y_lowest_ratio)

    save_scalers_y(y_highest_scalers, "highest", stockCode)

    # 학습 요청
    # train_ai_model(*make_lstm_dataset(x_volume_np, y_highest_np), stockCode, "volume", "highest")
    # train_ai_model(*make_lstm_dataset(x_investor_np, y_highest_np), stockCode, "investor", "highest")
    # train_ai_model(*make_lstm_dataset(x_short_np, y_highest_np), stockCode, "short", "highest")

    # train_ai_model(*make_lstm_dataset(x_volume_np, y_lowest_ratio_np), stockCode, "lowest")
    # train_lowest_investor_model(*make_lstm_dataset(x_investor_np, y_lowest_ratio_np))
    # train_lowest_short_model(*make_lstm_dataset(x_short_np, y_lowest_ratio_np))

    train_ai_model(*make_lstm_dataset(x_volume_np, y_is_up), stockCode, "volume", "is_up")
    # train_ai_model(*make_lstm_dataset(x_investor_np, y_is_up), stockCode, "investor", "is_up")
    # train_ai_model(*make_lstm_dataset(x_short_np, y_is_up), stockCode, "short", "is_up")

    # train_ai_model(*make_lstm_dataset(x_volume_np, y_is_down), stockCode, "is_down")
    # train_down_investor_model(make_lstm_dataset(x_investor_np, y_is_down))
    # train_down_short_model(make_lstm_dataset(x_short_np, y_is_down))

    print("전체 학습 완료")
    return

def inference(x_data, stockCode):

    x_volume, x_investor, x_short = [], [], []
    for row in x_data:
        base = list(row[0:6])                 # 시가·고가·저가·종가
        # x_volume.append(base + [row[5]] + list(row[15:20]) + [row[20]])        # + volume

        x_volume.append(base + list(row[15:]) + list(row[6:15])) # 시고저종, 거래량, 

        x_investor.append(base + list(row[6:10]))   # + 개인·외국인·기관·프로그램
        x_short.append(base + list(row[10:15]))     # + 공매도·기술지표

    # group_min/max  : 0~3열(가격군) 공통 스케일
    # other_scalers  : 4번 이후 각 열 독립 MinMaxScaler
    x_volume_scaler   = joblib.load(f"scalers/{stockCode}/X_volume.pkl")
    x_investor_scaler = joblib.load(f"scalers/{stockCode}/X_investor.pkl")
    x_short_scaler    = joblib.load(f"scalers/{stockCode}/X_short.pkl")

    #   shared_idx=[0,1,2,3] → OHLC 네 열만 공통 스케일러 사용
    x_volume_np = norm.normalize_2d_array(
        x_volume,
        shared_idx=[0, 1, 2, 3, 6, 7, 8, 9, 10],
        g_min=x_volume_scaler["group_min"],
        g_max=x_volume_scaler["group_max"],
        other_scalers=x_volume_scaler["other_scalers"]
    )[0]

    x_investor_np = norm.normalize_2d_array(
        x_investor,
        shared_idx=[0, 1, 2, 3],
        g_min=x_investor_scaler["group_min"],
        g_max=x_investor_scaler["group_max"],
        other_scalers=x_investor_scaler["other_scalers"]
    )[0]

    x_short_np = norm.normalize_2d_array(
        x_short,
        shared_idx=[0, 1, 2, 3],
        g_min=x_short_scaler["group_min"],
        g_max=x_short_scaler["group_max"],
        other_scalers=x_short_scaler["other_scalers"]
    )[0]

    # ───────────────── 모델 로드 ──────────────────
    #   ※ “is_up” 모델 사용. (highest 모델을 쓰려면 주석 변경)
    volume_model   = load_model(f"trained_model/{stockCode}/is_up_volume.h5")
    investor_model = load_model(f"trained_model/{stockCode}/is_up_investor.h5")
    short_model    = load_model(f"trained_model/{stockCode}/is_up_short.h5")

    # ───────────────── LSTM 시퀀스 변환 ──────────────
    x_volume_seq   = make_lstm_dataset(x_volume_np)[0]
    x_investor_seq = make_lstm_dataset(x_investor_np)[0]
    x_short_seq    = make_lstm_dataset(x_short_np)[0]

    # (batch=1, seq_len, feat) 형태로 reshape
    x_volume_ready   = x_volume_seq.reshape(1, x_volume_seq.shape[1], x_volume_seq.shape[2])
    x_investor_ready = x_investor_seq.reshape(1, x_investor_seq.shape[1], x_investor_seq.shape[2])
    x_short_ready    = x_short_seq.reshape(1, x_short_seq.shape[1], x_short_seq.shape[2])

    # ───────────────── 예측 ──────────────────────
    y_volume   = volume_model.predict(x_volume_ready,   verbose=1)
    print()
    # y_investor = investor_model.predict(x_investor_ready, verbose=1)
    print()
    # y_short    = short_model.predict(x_short_ready,    verbose=1)
    print()

    # ─── 스케일 되돌리기 전 raw 값 출력용 ─────────
    y_v = y_volume.ravel()[0]
    # y_i = y_investor.ravel()[0]
    # y_s = y_short.ravel()[0]
    print("raw volume  :", y_v)
    # print("raw investor:", y_i)
    # print("raw short   :", y_s)

    # ※ y_highest_scaler 등을 사용한 역정규화가 필요하다면
    #    denormalize_1d_array() 호출 부분을 주석 해제하고 사용
    # y_highest_scaler = joblib.load(f"scalers/{stockCode}/Y_highest.pkl")["scalers"]
    # y_v = norm.denormalize_1d_array([y_v], y_highest_scaler)[0]
    # y_i = norm.denormalize_1d_array([y_i], y_highest_scaler)[0]
    # y_s = norm.denormalize_1d_array([y_s], y_highest_scaler)[0]

    # ───────────────── 결과 합산 & 반환 ────────────
    # average = (y_v + y_i + y_s) / 3
    average = float(y_v)
    from flask import jsonify
    return jsonify(average)





def save_scalers_x(x_group_min, x_group_max, x_other_scalers, x_type, stockCode):
    save_obj = {
        "group_min": float(x_group_min),
        "group_max": float(x_group_max),
        "other_scalers": x_other_scalers
    }

    out_path = Path("scalers") / stockCode / f"X_{x_type}.pkl" # --------나중에 컨테이너로 올릴때 직접경로로 바꾸기----------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(save_obj, out_path)
    print(f"스케일 저장 완료 - {out_path}")

def save_scalers_y(y_scalers, y_type, stockCode):
    save_obj = {"scalers": y_scalers}

    out_path = Path("scalers") / stockCode / f"Y_{y_type}.pkl" # --------나중에 컨테이너로 올릴때 직접경로로 바꾸기---------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(save_obj, out_path)
    print(f"스케일 저장 완료 - {out_path}")
