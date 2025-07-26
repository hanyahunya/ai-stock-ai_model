from typing import Optional, Tuple, List, Dict
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

import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

def normalize_2d_array(
    arr_2d,
    shared_idx: List[int],  # 공통 스케일 (0~1)
    g_min: Optional[float] = None,
    g_max: Optional[float] = None,
    other_scalers: Optional[List[MinMaxScaler]] = None,
    minus1to1_idx: Optional[List[int]] = None,               # -1~1 정규화 대상
    minus1to1_params: Optional[Dict[int, Tuple[float, float]]] = None  # -1~1 스케일 파라미터
) -> Tuple[np.ndarray, Tuple[float, float], List[MinMaxScaler], Dict[int, Tuple[float, float]]]:
    arr_2d = np.asarray(arr_2d, dtype=np.float32)
    if minus1to1_idx is None:
        minus1to1_idx = []
    if minus1to1_params is None:
        minus1to1_params = {}

    # ── 1) NaN 제거 ───────────────────────────────
    mask_valid = ~np.isnan(arr_2d[:, shared_idx]).any(axis=1)
    arr_2d = arr_2d[mask_valid]

    # ── 2) 공통 스케일 (0~1) ──────────────────────
    shared_cols = arr_2d[:, shared_idx]
    if g_min is None or g_max is None:
        g_min = shared_cols.min()
        g_max = shared_cols.max()

    shared_norm = (shared_cols - g_min) / (g_max - g_min + 1e-9)

    # ── 3) -1~1 스케일 ────────────────────────────
    minus1to1_norm = {}
    updated_params = {}

    for idx in minus1to1_idx:
        col = arr_2d[:, idx]
        if idx in minus1to1_params:
            mid, scale = minus1to1_params[idx]
        else:
            mid = (col.max() + col.min()) / 2
            scale = (col.max() - col.min()) / 2 + 1e-9
        norm_col = (col - mid) / scale
        minus1to1_norm[idx] = norm_col
        updated_params[idx] = (mid, scale)

    # ── 4) 개별 MinMax 스케일 (0~1) ────────────────
    norm_cols, scalers = [], []
    other_idx = [
        i for i in range(arr_2d.shape[1])
        if i not in shared_idx and i not in minus1to1_idx
    ]

    if other_scalers is None:
        for idx in other_idx:
            sc = MinMaxScaler()
            col_norm = sc.fit_transform(arr_2d[:, idx].reshape(-1, 1))
            norm_cols.append(col_norm)
            scalers.append(sc)
    else:
        if len(other_scalers) != len(other_idx):
            raise ValueError("스케일러 개수가 맞지 않습니다.")
        for idx, sc in zip(other_idx, other_scalers):
            norm_cols.append(sc.transform(arr_2d[:, idx].reshape(-1, 1)))
        scalers = other_scalers

    # ── 5) 병합 (원래 순서 유지) ───────────────────
    normalized = np.empty_like(arr_2d, dtype=np.float32)
    normalized[:, shared_idx] = shared_norm
    normalized[:, other_idx] = np.hstack(norm_cols)
    for idx in minus1to1_idx:
        normalized[:, idx] = minus1to1_norm[idx]

    return normalized, (g_min, g_max), scalers, updated_params


def denormalize_2d_array(
    norm_arr_2d,
    shared_idx: List[int],
    group_min_max: Tuple[float, float],
    other_scalers: List[MinMaxScaler],
    minus1to1_idx: Optional[List[int]] = None,
    minus1to1_params: Optional[dict] = None
) -> np.ndarray:
    norm_arr_2d = np.asarray(norm_arr_2d, dtype=np.float32)
    g_min, g_max = group_min_max
    if minus1to1_idx is None:
        minus1to1_idx = []
    if minus1to1_params is None:
        minus1to1_params = {}

    # ── 1) 공통 스케일 역변환 ──────────────────────
    shared_denorm = norm_arr_2d[:, shared_idx] * (g_max - g_min) + g_min

    # ── 2) 개별 스케일 역변환 ──────────────────────
    other_idx = [
        i for i in range(norm_arr_2d.shape[1])
        if i not in shared_idx and i not in minus1to1_idx
    ]
    restored_cols = []
    for idx, sc in zip(other_idx, other_scalers):
        col = norm_arr_2d[:, idx].reshape(-1, 1)
        restored_cols.append(sc.inverse_transform(col))

    # ── 3) -1~1 역변환 ────────────────────────────
    minus1to1_restored = {}
    for idx in minus1to1_idx:
        mid, scale = minus1to1_params[idx]
        col = norm_arr_2d[:, idx]
        minus1to1_restored[idx] = col * scale + mid

    # ── 4) 병합 (원래 순서 유지) ───────────────────
    restored_all = np.empty_like(norm_arr_2d, dtype=np.float32)
    restored_all[:, shared_idx] = shared_denorm
    restored_all[:, other_idx] = np.hstack(restored_cols)
    for idx in minus1to1_idx:
        restored_all[:, idx] = minus1to1_restored[idx]

    return restored_all











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

def normalize_2d_array1(
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


def denormalize_2d_array1(norm_arr_2d, group_min_max, other_scalers):
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



# ───────── 정규화 ─────────
def normalize_2d_array2(
    arr_2d,
    shared_idx: List[int],                       # 공통 스케일 열
    g_min: Optional[float] = None,
    g_max: Optional[float] = None,
    other_scalers: Optional[List[MinMaxScaler]] = None,
):
    """
    - shared_idx 열 가운데 NaN 이 하나라도 있으면 해당 행을 DROP.
    - 그 뒤 공통 스케일러(OHLC·MA 등)와 개별 스케일러를 적용.
    """
    arr_2d = np.asarray(arr_2d, dtype=np.float32)

    # ── 1) NaN 행 제거 ───────────────────────────────────
    mask_valid = ~np.isnan(arr_2d[:, shared_idx]).any(axis=1)
    arr_2d = arr_2d[mask_valid]                  # NaN 포함 행이 모두 사라짐

    # ── 2) 공통 스케일 ───────────────────────────────────
    shared_cols = arr_2d[:, shared_idx]

    if g_min is None or g_max is None:           # fit
        g_min = shared_cols.min()
        g_max = shared_cols.max()

    shared_norm = (shared_cols - g_min) / (g_max - g_min + 1e-9)

    # ── 3) 개별 스케일 ───────────────────────────────────
    other_idx = [i for i in range(arr_2d.shape[1]) if i not in shared_idx]
    norm_cols, scalers = [], []

    if other_scalers is None:                    # fit
        for idx in other_idx:
            sc = MinMaxScaler()
            col_norm = sc.fit_transform(arr_2d[:, idx].reshape(-1, 1))
            norm_cols.append(col_norm)
            scalers.append(sc)
    else:                                        # transform
        if len(other_scalers) != len(other_idx):
            raise ValueError("스케일러 개수가 맞지 않습니다.")
        for idx, sc in zip(other_idx, other_scalers):
            norm_cols.append(sc.transform(arr_2d[:, idx].reshape(-1, 1)))
        scalers = other_scalers

    # ── 4) 병합(원래 순서 유지) ───────────────────────────
    normalized = np.empty_like(arr_2d, dtype=np.float32)
    normalized[:, shared_idx] = shared_norm
    normalized[:, other_idx]  = np.hstack(norm_cols)

    return normalized, (g_min, g_max), scalers



# ───────── 역-정규화 ─────────
def denormalize_2d_array2(
    norm_arr_2d,
    shared_idx: List[int],
    group_min_max: Tuple[float, float],
    other_scalers: List[MinMaxScaler],
):
    norm_arr_2d = np.asarray(norm_arr_2d, dtype=np.float32)
    g_min, g_max = group_min_max

    # ── 공통 스케일 역변환
    shared_denorm = norm_arr_2d[:, shared_idx] * (g_max - g_min) + g_min

    # ── 개별 스케일 역변환
    other_idx = [i for i in range(norm_arr_2d.shape[1]) if i not in shared_idx]
    restored_cols = []
    for idx, sc in zip(other_idx, other_scalers):
        col = norm_arr_2d[:, idx].reshape(-1, 1)
        restored_cols.append(sc.inverse_transform(col))

    # ── 병합(원래 순서 유지)
    restored_all = np.empty_like(norm_arr_2d, dtype=np.float32)
    restored_all[:, shared_idx] = shared_denorm
    restored_all[:, other_idx]  = np.hstack(restored_cols)

    return restored_all


def normalize_2d_array33(
    arr_2d,
    shared_idx: List[int],  # 공통 스케일 (0~1)
    g_min: Optional[float] = None,
    g_max: Optional[float] = None,
    other_scalers: Optional[List[MinMaxScaler]] = None,
    minus1to1_idx: Optional[List[int]] = None  # -1~1 정규화 대상
) -> Tuple[np.ndarray, Tuple[float, float], List[MinMaxScaler], dict]:
    arr_2d = np.asarray(arr_2d, dtype=np.float32)
    if minus1to1_idx is None:
        minus1to1_idx = []

    # ── 1) NaN 제거 ───────────────────────────────
    mask_valid = ~np.isnan(arr_2d[:, shared_idx]).any(axis=1)
    arr_2d = arr_2d[mask_valid]

    # ── 2) 공통 스케일 (0~1) ──────────────────────
    shared_cols = arr_2d[:, shared_idx]
    if g_min is None or g_max is None:
        g_min = shared_cols.min()
        g_max = shared_cols.max()

    shared_norm = (shared_cols - g_min) / (g_max - g_min + 1e-9)

    # ── 3) -1~1 스케일 ────────────────────────────
    minus1to1_norm = {}
    minus1to1_params = {}
    for idx in minus1to1_idx:
        col = arr_2d[:, idx]
        mid = (col.max() + col.min()) / 2
        scale = (col.max() - col.min()) / 2 + 1e-9
        norm_col = (col - mid) / scale
        minus1to1_norm[idx] = norm_col
        minus1to1_params[idx] = (mid, scale)

    # ── 4) 개별 MinMax 스케일 (0~1) ────────────────
    norm_cols, scalers = [], []
    other_idx = [
        i for i in range(arr_2d.shape[1])
        if i not in shared_idx and i not in minus1to1_idx
    ]

    if other_scalers is None:
        for idx in other_idx:
            sc = MinMaxScaler()
            col_norm = sc.fit_transform(arr_2d[:, idx].reshape(-1, 1))
            norm_cols.append(col_norm)
            scalers.append(sc)
    else:
        if len(other_scalers) != len(other_idx):
            raise ValueError("스케일러 개수가 맞지 않습니다.")
        for idx, sc in zip(other_idx, other_scalers):
            norm_cols.append(sc.transform(arr_2d[:, idx].reshape(-1, 1)))
        scalers = other_scalers

    # ── 5) 병합 (원래 순서 유지) ───────────────────
    normalized = np.empty_like(arr_2d, dtype=np.float32)
    normalized[:, shared_idx] = shared_norm
    normalized[:, other_idx] = np.hstack(norm_cols)
    for idx in minus1to1_idx:
        normalized[:, idx] = minus1to1_norm[idx]

    return normalized, (g_min, g_max), scalers, minus1to1_params
