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

def make_lstm_dataset(
    X_raw: np.ndarray,
    y_raw: Optional[np.ndarray] = None,
    sequence_length: int = 45
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
        base = list(row[0:4])  # 시가, 고가, 저가, 종가, #등락률(0:5)

        x_v = base + [row[5]]                  # + volume
        x_i = base + list(row[6:10])           # + individual, fore, institution, program
        x_s = base + list(row[10:15])          # + 공매도 및 기술지표

        x_volume.append(x_v)
        x_investor.append(x_i)
        x_short.append(x_s)

    x_volume_np, (x_volume_group_min, x_volume_group_max), x_volume_other_scalers = norm.normalize_2d_array(x_volume)
    x_investor_np, (x_investor_group_min, x_investor_group_max), x_investor_other_scalers =  norm.normalize_2d_array(x_investor)
    x_short_np, (x_short_group_min, x_short_group_max), x_short_other_scalers = norm.normalize_2d_array(x_short)

    save_scalers_x(x_volume_group_min, x_volume_group_max, x_volume_other_scalers, "volume", stockCode)
    save_scalers_x(x_investor_group_min, x_investor_group_max, x_investor_other_scalers, "investor", stockCode)
    save_scalers_x(x_short_group_min, x_short_group_max, x_short_other_scalers, "short", stockCode)


    # inference(x_volume, stockCode, x_volume_np)

    # Y 데이터 분리
    y_highest_ratio = [row[0] for row in y_data]
    y_lowest_ratio = [row[1] for row in y_data]
    y_is_up = [row[2] for row in y_data]
    y_is_down = [row[3] for row in y_data]

    y_highest_np, y_highest_scalers = norm.normalize_1d_array(y_highest_ratio)
    y_lowest_np, y_lowest_scalers = norm.normalize_1d_array(y_lowest_ratio)

    save_scalers_y(y_highest_scalers, "highest", stockCode)

    # 학습 요청
    # train_ai_model(*make_lstm_dataset(x_volume_np, y_highest_np), stockCode, "volume", "highest")
    # train_ai_model(*make_lstm_dataset(x_investor_np, y_highest_np), stockCode, "investor", "highest")
    train_ai_model(*make_lstm_dataset(x_short_np, y_highest_np), stockCode, "short", "highest")

    # train_ai_model(*make_lstm_dataset(x_volume_np, y_lowest_ratio_np), stockCode, "lowest")
    # train_lowest_investor_model(*make_lstm_dataset(x_investor_np, y_lowest_ratio_np))
    # train_lowest_short_model(*make_lstm_dataset(x_short_np, y_lowest_ratio_np))

    # train_ai_model(*make_lstm_dataset(x_volume_np, y_is_up), stockCode, "volume", "is_up")
    # train_ai_model(*make_lstm_dataset(x_investor_np, y_is_up), stockCode, "investor", "is_up")
    # train_ai_model(*make_lstm_dataset(x_short_np, y_is_up), stockCode, "short", "is_up")

    # train_ai_model(*make_lstm_dataset(x_volume_np, y_is_down), stockCode, "is_down")
    # train_down_investor_model(make_lstm_dataset(x_investor_np, y_is_down))
    # train_down_short_model(make_lstm_dataset(x_short_np, y_is_down))

    print("전체 학습 완료")
    return

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

def inference(x_data, stockCode):

    # 학습시와 같은 배열로 끊기
    x_volume, x_investor, x_short = [], [], []
    for row in x_data:
        base = list(row[0:4])  # 시가, 고가, 저가, 종가, #등락률0:5

        x_v = base + [row[5]]                  # + volume
        x_i = base + list(row[6:10])           # + individual, fore, institution, program
        x_s = base + list(row[10:15])          # + 공매도 및 기술지표
        x_volume.append(x_v)
        x_investor.append(x_i)
        x_short.append(x_s)

    # 학습 모델의 마지막에 사용된 x데이터의 스케일 불러오기
    x_volume_scaler = joblib.load("scalers/"+stockCode+"/X_volume.pkl")
    x_volume_g_min, x_volume_g_max = x_volume_scaler["group_min"], x_volume_scaler["group_max"]
    x_volume_other_scaler = x_volume_scaler["other_scalers"]

    x_investor_scaler = joblib.load("scalers/"+stockCode+"/X_investor.pkl")
    x_investor_g_min, x_investor_g_max = x_investor_scaler["group_min"], x_investor_scaler["group_max"]
    x_investor_other_scaler = x_investor_scaler["other_scalers"]

    x_short_scaler = joblib.load("scalers/"+stockCode+"/X_short.pkl")
    x_short_g_min, x_short_g_max = x_short_scaler["group_min"], x_short_scaler["group_max"]
    x_short_other_scaler = x_short_scaler["other_scalers"]
    # y데이터의 스케일
    y_highest_scaler_data = joblib.load("scalers/"+stockCode+"/Y_highest.pkl")
    y_highest_scaler = y_highest_scaler_data["scalers"]

    # 이전 학습한 모델의 스케일로 정규화된 데이터
    x_volume_np = norm.normalize_2d_array(x_volume, x_volume_g_min, x_volume_g_max, x_volume_other_scaler)[0]
    x_investor_np = norm.normalize_2d_array(x_investor, x_investor_g_min, x_investor_g_max, x_investor_other_scaler)[0]
    x_short_np = norm.normalize_2d_array(x_short, x_short_g_min, x_short_g_max, x_short_other_scaler)[0]

    volume_model = load_model("trained_model/"+stockCode+"/highest_volume.h5")
    investor_model = load_model("trained_model/"+stockCode+"/highest_investor.h5")
    short_model = load_model("trained_model/"+stockCode+"/highest_short.h5")

    # volume_model = load_model("trained_model/"+stockCode+"/is_up_volume.h5")
    # investor_model = load_model("trained_model/"+stockCode+"/is_up_investor.h5")
    # short_model = load_model("trained_model/"+stockCode+"/is_up_short.h5")

    x_volume_data = make_lstm_dataset(x_volume_np)[0]
    x_investor_data = make_lstm_dataset(x_investor_np)[0]
    x_short_data = make_lstm_dataset(x_short_np)[0]

    # x_volume_ready = x_volume_data.reshape(1, x_volume_data.shape[1], x_volume_data.shape[2])
    # x_investor_ready = x_investor_data.reshape(1, x_investor_data.shape[1], x_investor_data.shape[2])
    x_short_ready = x_short_data.reshape(1, x_short_data.shape[1], x_short_data.shape[2])

    # y_volume = volume_model.predict(x_volume_ready, verbose = 1)
    print("\n\n")
    # y_investor = investor_model.predict(x_investor_ready, verbose = 1)
    # print("\n\n")
    y_short = short_model.predict(x_short_ready, verbose = 1)
    print("\n\n")
    print("\n\n")

    # y1 = norm.denormalize_1d_array([y_volume.ravel()[0]], y_highest_scaler)
    # y2 = norm.denormalize_1d_array([y_investor.ravel()[0]], y_highest_scaler)
    y3 = norm.denormalize_1d_array([y_short.ravel()[0]], y_highest_scaler)

    # print(f"volume   {y1}")
    # print(f"investor {y2}")
    print(f"short    {y3}")

    # print(f"volume   {y_volume}")
    # print(f"investor {y_investor}")
    # print(f"short    {y_short}")

    del volume_model                                # 파이썬 레벨에서 참조 제거
    tf.keras.backend.clear_session()         # 그래프·GPU 메모리 해제

    return













def train_model1(x_data, y_data, stockCode):
    # X 데이터 분리
    x_volume, x_investor, x_short = [], [], []
    for row in x_data:
        base = list(row[0:5])  # 시가, 고가, 저가, 종가, #등락률(0:5)

        x_v = base + list(row[5:10])                # + volume

        x_volume.append(x_v)

    

    x_volume_np, (x_volume_group_min, x_volume_group_max), x_volume_other_scalers = norm.normalize_2d_array(x_volume)

    save_scalers_x(x_volume_group_min, x_volume_group_max, x_volume_other_scalers, "volume", stockCode)

    # inference(x_volume, stockCode, x_volume_np)

    # Y 데이터 분리
    y_highest_ratio = [row[0] for row in y_data]
    y_lowest_ratio = [row[1] for row in y_data]
    y_is_up = [row[2] for row in y_data]
    y_is_down = [row[3] for row in y_data]

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


    # train_ai_model(*make_lstm_dataset(x_volume_np, y_is_down), stockCode, "is_down")
    # train_down_investor_model(make_lstm_dataset(x_investor_np, y_is_down))
    # train_down_short_model(make_lstm_dataset(x_short_np, y_is_down))

    print("전체 학습 완료")
    return

def inference1(x_data, stockCode):

    # 학습시와 같은 배열로 끊기
    x_volume, x_investor, x_short = [], [], []
    for row in x_data:
        base = list(row[0:4])  # 시가, 고가, 저가, 종가, #등락률0:5

        x_v = base + [row[5:15]]                  # + volume
     
        x_volume.append(x_v)
    

    # 학습 모델의 마지막에 사용된 x데이터의 스케일 불러오기
    x_volume_scaler = joblib.load("scalers/"+stockCode+"/X_volume.pkl")
    x_volume_g_min, x_volume_g_max = x_volume_scaler["group_min"], x_volume_scaler["group_max"]
    x_volume_other_scaler = x_volume_scaler["other_scalers"]

   
    # y데이터의 스케일
    y_highest_scaler_data = joblib.load("scalers/"+stockCode+"/Y_highest.pkl")
    y_highest_scaler = y_highest_scaler_data["scalers"]

    # 이전 학습한 모델의 스케일로 정규화된 데이터
    x_volume_np = norm.normalize_2d_array(x_volume, x_volume_g_min, x_volume_g_max, x_volume_other_scaler)[0]
    
    # volume_model = load_model("trained_model/"+stockCode+"/highest_volume.h5")
    # investor_model = load_model("trained_model/"+stockCode+"/highest_investor.h5")
    # short_model = load_model("trained_model/"+stockCode+"/highest_short.h5")

    volume_model = load_model("trained_model/"+stockCode+"/is_up_volume.h5")
   

    x_volume_data = make_lstm_dataset(x_volume_np)[0]
    
    x_volume_ready = x_volume_data.reshape(1, x_volume_data.shape[1], x_volume_data.shape[2])
   
    y_volume = volume_model.predict(x_volume_ready, verbose = 1)
    print("\n\n")
   

    # y1 = norm.denormalize_1d_array([y_volume.ravel()[0]], y_highest_scaler)
    # y2 = norm.denormalize_1d_array([y_investor.ravel()[0]], y_highest_scaler)
    # y3 = norm.denormalize_1d_array([y_short.ravel()[0]], y_highest_scaler)

    # print(f"volume   {y1}")
    # print(f"investor {y2}")
    # print(f"short    {y3}")

    print(f"volume   {y_volume}")


    return




def debug_rows(arr_2d, expected_len):
    """행 길이·원소 타입을 검사해 문제가 있는 행을 출력"""
    import math, numbers
    for idx, row in enumerate(arr_2d):
        # ① 길이 체크
        if len(row) != expected_len:
            print(f"[LEN] idx={idx}  len={len(row)}  row={row}")
            continue

        # ② 각 원소가 스칼라인지 체크
        bad = [
            (j, v) for j, v in enumerate(row)
            if not (isinstance(v, numbers.Number) and not isinstance(v, bool))
            or (isinstance(v, float) and math.isnan(v))
        ]
        if bad:
            print(f"[TYPE] idx={idx}  bad={bad}  row={row}")