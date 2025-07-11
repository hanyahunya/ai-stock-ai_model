# Yデータ変更（仮）

> **目的**  
> 推測の難易度が高い回帰データを予測する前に、まず二値分類で精度を向上させて予測

---

## 概要
| 項目      | 変更前                            | 変更後                                         |
| --------- | --------------------------------- | ---------------------------------------------- |
| **Yデータ** | 現在値比7日以内の最高収益率(%)        | 現在値比7日以内に3％以上上昇したかどうか (0,1) |
| **損失関数** | Huber / MSE                     | BCE                                            |

---

## 変更理由
* 回帰予測に用いていたYデータが-3％〜3％に集中しており、予測時に平均値付近しか予測しない状況が発生  
  -> 解決したかったが実力不足のため、まずはYデータを簡単な二値分類にして開発し、その後で回帰モデルの開発に着手する予定。  
* Xデータにテクニカル指標を追加する予定だが、予測精度の変化を把握しやすいように、データをシンプルな二値分類とするためYデータを選定

---

## 今後追加予定のXデータ
* MA (Moving Average) ‐ 移動平均線  
  -> 一定期間の平均価格 (5, 10, 20, 60, 120)
* EMA (Exponential Moving Average) ‐ 指数移動平均線  
  -> 最新価格により大きな重み
* MACD
* RSI ‐ 相対力指数
* Bollinger Bands
* など…（多数）  
> 各指標をX_data配列に追加する関数を.pyファイルごとに分割する予定
---


<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>


# Y데이터 변경 (임시)

> **목적**  
> 추측 난이도가 어려운 회귀데이터를 예측하기전에, 이진분류로 정확도를 개선하여 예측

---

## 개요
| 항목      | 변경 전        | 변경후                      |
| --------- | --------- | ---------------------------- |
| **Y데이터** | 현재가 대비 7일이내 최고수익률(%) | 현재가 대비 7일이내 3%이상 상승여부 (0,1) |
| **손실함수** | Huber / MSE | BCE

---

## 바꾸게 된 이유
* 회귀예측에 사용되는 Y데이터가 -3% ~ 3%에 몰려있어서 예측시 평균값 근처만 예측하는 상황 발생  
-> 해결하고 싶었지만, 실력부족으로 우선 Y데이터가 간단한 이진분류 먼저 개발후 회귀모델 개발 착수 예정.
* X데이터에 기술적 지표들을 추가할 예정인데, 예측 정확도의 변화를 알아볼수있게 데이터가 간단한 이진분류로 Y데이터를 선택

---

## 이후 추가할 X데이터
* MA (Moving Average) - 이동평균선  
-> 일정기간 평균가격 (5, 10, 20, 60, 120)
* EMA (Exponential Moving Average) - 지수이동평균선  
-> 최근 가격에 더 큰 가중치
* MACD
* RSI - 상대강도지수
* Bollinger Bands
* 등등...(너무 많음)
> 각 지표를 X_data배열에 추가하는 함수를 py파일별로 나눌예정
---

