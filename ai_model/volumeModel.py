import numpy as np
import os
import datetime
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import Huber
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from dotenv import load_dotenv
load_dotenv()

def train_ai_model(X, Y, stockCode, x_type, y_type, epochs=2000, validation_split=0.2):

    if y_type not in ["highest", "lowest", "is_up", "is_down"]:
        return

    is_Y_binary = True
    if (y_type in ["highest", "lowest"]):
        is_Y_binary = False

    # 환경변수에서 lstm 설정 값 로드 (Json형식)
    if (is_Y_binary):
        params_json = os.environ.get("AI_TRAIN_PARAMS_BINARY")
    else:
        params_json = os.environ.get("AI_TRAIN_PARAMS")

    if not params_json:
        print("환경변수가 설정되지 않았습니다.")
        return

    params = json.loads(params_json)
    n_layers = params["layers"] # 레이어수 int
    learning_rate = params["learning_rate"] # 학습률 0.00001 같은거. <- 경사하강법에서 기울기에 얼마나 곱해서 이동시킬지
    batch_size = params["batch_size"] # 배치 <- 1000개의 X 시계열데이터가 있을때, 배치 수만큼 묶어서 학습


    print("\n\n")
    print(params)
    print("\n\n")

    # tensorboard 로그 경로 설정 <- tensorboard --logdir train_logs/005930/highest/2025-06-24_23-06 이런식으로 과거 학습데이터 열람 가능 
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = os.path.join("train_logs", stockCode, y_type, x_type, timestamp) # --------나중에 컨테이너로 올릴때 직접경로로 바꾸기---------
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    if (is_Y_binary):
        l2_reg = regularizers.l2(1e-4)  #-----------------------------L2정규화 이진분류-------------------
    else:
        l2_reg = regularizers.l2(1e-4)  #-----------------------------L2정규화 회귀-------------------

    # ai모델
    model = Sequential()
    for i in range(n_layers):
        units = params[f"nodes_layer{i+1}"]
        dropout_rate = params[f"dropout_l{i}"] #-----드롭아웃끔 나중에 과적합일때 다시키쇼-----<과적합>------------------
        return_seq = (i < n_layers - 1) # 만약 지금이 마지막 레이어면 false -> lstm에서 다음 레이어에 시퀀스 전달안함
        if i == 0:
            # 첫번째 lstm레이어에선 입력차원( X데이터를 넣을 텐서의 모양 )에 (batch수, 시계열길이(60일), 피처수(open_price, close_price, volume, ...))
            # 를 넣는데 batch수는 나중에 model.fit에서 설정가능.
            # model.add(LSTM(units=units, return_sequences=return_seq, input_shape=(X.shape[1], X.shape[2])))
            model.add(LSTM(units=units, return_sequences=return_seq, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2_reg, bias_regularizer=l2_reg))  #-----------------------------L2정규화-------------------
        else:
            #두번째 레이어 이상부턴 x텐서모양 설정 안해줘도됨
            # model.add(LSTM(units=units, return_sequences=return_seq))
            model.add(LSTM(units=units, return_sequences=return_seq, kernel_regularizer=l2_reg, bias_regularizer=l2_reg)) #-----------------------------L2정규화-------------------
        # 한레이어 뒤에 과적합 방지를 위해 드롭아웃 레이어 추가. **드롭아웃 안붙히거나 환경변수에서 레이어에 맞게 드롭아웃값 안넣어줬을땐 로직 변경 필요**(중요)

        model.add(Dropout(dropout_rate)) # --------------------------------드롭아웃----------------------------------

    # 옵티마이저 경사하강법 그거. 손실함수의 값을 작게하는게 옵티마이저
    #
    # SGD <- 기본적인 경사하강법. 한번에 하나의 미니배치로 기울기를 계산, 가중치를 조금씩 줄임
    # 장점: 구조 단순, 해석 쉬움 / 단점: 학습 느리고 진동 많음
    #
    # SGD + Momentum <- 일반 SGD는 가끔 지그재그로 움직이는데, 이전에 가던방향을기억, 더 부드럽게 이동해줌. (관성느낌)        이미지분야, 일반 딥러닝
    # 장점: 진동 줄이고 빠른 수렴 / 단점: 여전히 학습률 튜닝 필요
    # 사용예시 SGD(learning_rate=0.01, momentum=0.9)
    #
    # RMSprop <- 최근 기울기의 제곲값을 평균내서 학습률 자동조정. 기울기가 자주튀는 방향은 천천히, 안정적인방향은 빨리 움직임.   LSTM, 시계열 데이터
    # 장점: LSTM/RNN 학습에 강함 / 단점: 성능 튀는 경우 있음 (일관성 적음)
    #
    # Adam <- Momentom + RMSprop 느낌. 거의 모든상황에서 안정적                                                       그냥 대부분 이거씀                              
    # 장점: 튜닝 없이도 잘 됨, 거의 모든 모델에서 기본 / 단점: 때때로 일반화(검증성능)는 SGD보다 약간 떨어짐
    optimizer = RMSprop(learning_rate=learning_rate)
    # optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

    if (is_Y_binary):
        # 예측할것이 1일 확률 하나이기에 1개의 출력층, sigmoid로 0~1로 변환
        model.add(Dense(1, activation='sigmoid'))
        # 이진분류용 loss(손실함수)
        # binary_crossentropy <- y값이 0, 1일때 사용. 로그기반이라 예측이 틀릴수록 손실을 크게줌.
        model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                      metrics=[
                          'accuracy', tf.keras.metrics.AUC(name="auc"), 
                          tf.keras.metrics.Precision(thresholds=0.4, name="prec"), 
                          tf.keras.metrics.Recall(thresholds=0.4, name="recall"),
                      ]
        ) # metrics는 그냥 보여주기용, 역전파안함(학습안함)
    else:
        model.add(Dense(1))
        # 회귀용 loss(손실함수)
        # MSE <- 평균제곱오차, 예측오차를 제곱해서 평균냄.
        # - 큰 오차에 민감 <- 이상치가 있으면 손실이 엄청 커짐
        # - 학습이 부드럽고 안정적 (미분이 쉬움)
        #
        # MAE <- 평균 절댓값 오차, 오차의 절댓값을 평균냄.
        # - 큰 오차(이상치)에 덜민감
        # - 단점: 미분이 불연속(절댓값은 0에서 미분이 뚝 끊김), 학습이 덜 안정적일수 있음
        # - 이상치가 많을때
        #
        # Hubur Loss <- MSE, MAE의 장점만 합친 절충형 손실함수
        # - 작은오차엔 MSE처럼 민감하게, 큰오차엔 MAE처럼 덜민감하게
        # - MSE처럼 미분이 부드러움. (경사하강법에 적합)
        # - 파라미터로 delta 요구 (MSE / MAE로 전환되는 경계값)
        ## delta
        ## 예측오차가 +-1 내외 -> delta=1.0(default)써도 무난
        ## 오차가 엄청 커질수 있음 -> 오차분포의 표준편차정도를 쓰면 된다네요
        ## 이상치가 많아서 큰오차를 덜 민감하게 -> delta를 작게 (예:0.5)
        ## 이상치가 거의없고 세밀한 차이를 강조하고싶을떄 -> delta를 크게 (예:5.0)   
        # model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        loss_fn = Huber(delta=0.1)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mse'])
    
    # 다중 클래스 분류용 손실함수는 categorical_crossentropy / sparse_categorical_crossentropy 가 있다네요

    
    # ------스케쥴-----------
    lr_sched = ReduceLROnPlateau(
        monitor='val_loss',   # 감시 대상
        factor=0.5,           # 개선 없으면 학습률 × 0.5
        patience=5,           # 5에포크 연속 개선 없을 때 발동
        min_lr=1e-6           # 최소 LR 한계
    )


    # 학습
    try:
        model.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            validation_split=validation_split,
            callbacks=[lr_sched, StopAfterMinLR(min_lr=1e-6, wait=20), tensorboard_callback], # 스케쥴 추가  localhost:6006 에서 학습결과 볼수있게
            verbose=1
        )
    except KeyboardInterrupt:
        print("\n\n학습중단")
    # 모델 저장
    save_dir = os.path.join("trained_model", stockCode) # --------나중에 컨테이너로 올릴때 직접경로로 바꾸기---------
    os.makedirs(save_dir, exist_ok=True)
    # trained_model/주식코드/y타입.h5 (예: trained_model/005930/highest.h5)
    model_path = os.path.join(save_dir, y_type + "_" + x_type + ".h5")
    model.save(model_path)
    print(f"모델 저장 완료: {model_path}")

    return model




class StopAfterMinLR(Callback):
    def __init__(self, min_lr=1e-6, wait=20):
        super().__init__()
        self.min_lr = min_lr
        self.wait = wait
        self.hit = None

    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.lr.numpy())
        if lr <= self.min_lr:
            if self.hit is None:
                self.hit = epoch
            elif epoch - self.hit >= self.wait:
                self.model.stop_training = True