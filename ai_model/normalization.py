from typing import Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1d for normalize highest, lowest y_value
def normalize_1d_array(arr):
    arr_np = np.array(arr).reshape(-1, 1)  # 
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(arr_np).ravel()  # 1차원 배열로 다시 변환
    return normalized, scaler

def denormalize_1d_array(normalized_arr, scaler):

    arr_np = np.array(normalized_arr).reshape(-1, 1)
    original = scaler.inverse_transform(arr_np).ravel()
    return original
# -------------------2d---------------------

def normalize_2d_array_old_version(arr_2d):
    arr_2d = np.array(arr_2d)  # (n_samples, n_features)

    # 0~3번 인덱스 → 같은 min, max로 정규화
    group = arr_2d[:, 0:4]  # 시가, 고가, 저가, 종가
    group_min = group.min()
    group_max = group.max()
    group_0_3 = (group - group_min) / (group_max - group_min)

    # 나머지 각 피처 → 따로 정규화
    other_cols = []
    other_scalers = []
    for i in range(4, arr_2d.shape[1]):
        scaler = MinMaxScaler()
        col = arr_2d[:, i].reshape(-1, 1)  # (n, 1)
        norm_col = scaler.fit_transform(col)
        other_cols.append(norm_col)  # list of (n, 1)
        other_scalers.append(scaler)

    # 수평 연결 (n, 4) + (n, ?)
    other_group = np.hstack(other_cols)
    normalized_all = np.hstack([group_0_3, other_group])

    return normalized_all, (group_min, group_max), other_scalers

def normalize_2d_array(
    arr_2d,
    x_group_min: Optional[float] = None,
    x_group_max: Optional[float] = None,
    x_other_scalers: Optional[List[MinMaxScaler]] = None,
) -> Tuple[np.ndarray, Tuple[float, float], List[MinMaxScaler]]:

    arr_2d = np.asarray(arr_2d)

    # --------------- 0~3열 (공통 스케일) ---------------------
    if x_group_min is None or x_group_max is None:
        g_min, g_max = arr_2d[:, :4].min(), arr_2d[:, :4].max()
    else:
        g_min, g_max = x_group_min, x_group_max

    group_norm = (arr_2d[:, :4] - g_min) / (g_max - g_min + 1e-9)

    # ---------------- 4열 이후 (개별 스케일) ------------------
    norm_cols, scalers = [], []
    if x_other_scalers is None:
        # fit 모드
        for i in range(4, arr_2d.shape[1]):
            sc = MinMaxScaler()
            col_norm = sc.fit_transform(arr_2d[:, i].reshape(-1, 1))
            norm_cols.append(col_norm)
            scalers.append(sc)
    else:
        # transform 모드
        if len(x_other_scalers) != arr_2d.shape[1] - 4:
            raise ValueError("스케일러 개수가 피처 수와 일치하지 않습니다.")
        for i, sc in enumerate(x_other_scalers, start=4):
            col_norm = sc.transform(arr_2d[:, i].reshape(-1, 1))
            norm_cols.append(col_norm)
        scalers = x_other_scalers

    other_norm = np.hstack(norm_cols)
    normalized = np.hstack([group_norm, other_norm])

    return normalized, (g_min, g_max), scalers


def denormalize_2d_array(norm_arr_2d, group_min_max, other_scalers):
    norm_arr_2d = np.array(norm_arr_2d)
    group_min, group_max = group_min_max

    # 0~3번 복원
    orig_0_3 = norm_arr_2d[:, 0:4] * (group_max - group_min) + group_min

    # 나머지 피처들 복원
    restored_cols = []
    for i, scaler in enumerate(other_scalers):
        col = norm_arr_2d[:, 4 + i].reshape(-1, 1)
        restored = scaler.inverse_transform(col)
        restored_cols.append(restored)

    # 병합
    restored_rest = np.hstack(restored_cols)
    restored_all = np.hstack([orig_0_3, restored_rest])

    return restored_all

