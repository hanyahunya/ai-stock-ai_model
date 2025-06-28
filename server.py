from flask import Flask, jsonify
import mysql.connector
import os
from threading import Thread
from ai_model.router import train_model, inference

app = Flask(__name__)

def fetch_and_run(stockCode):
    print("백그라운드 작업 시작")

    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user=os.environ["AI_STOCK_DB_USERNAME"],
        password=os.environ["AI_STOCK_DB_PASSWORD"],
        database="ai_stock"
    )
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT open_price, high_price, low_price, close_price, price_change_rate,
               volume, individual, fore, institution, program,
               shrts_qty, ovr_shrts_qty, trde_wght, shrts_trde_price, shrts_avg_pric,
               highest_ratio_7_days, lowest_ratio_7_days,
               is_up_3_percent_in_7_days, is_down_3_percent_in_7_days
        FROM daily_stock
        WHERE stock_code = %s and date > '2006-01-01'
        ORDER BY date ASC
        """,
        #and date > '2010-01-01'
        (stockCode,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    rows = rows[:-6] # 최근 6일데이터는 y값이 없기에 학습에서 제외
    

    x_data = [row[0:15] for row in rows]

    y_data = [row[15:] for row in rows]

    train_model(x_data, y_data, stockCode)
    print("백그라운드 작업 완료")

def inference_run(stockCode, date):
    print("백그라운드 작업 시작")

    conn = mysql.connector.connect(
        host="localhost",
        port=3306,
        user=os.environ["AI_STOCK_DB_USERNAME"],
        password=os.environ["AI_STOCK_DB_PASSWORD"],
        database="ai_stock"
    )
    cursor = conn.cursor()

    # 학습을 과거 -> 현재순의 시계열데이터로 학습했기에, 적용할 데이터도 최근 60일의 데이터를 끊고, 불러온 데이터를 과거순으로 다시 정렬
    cursor.execute(
        """
        SELECT  *
        FROM (
            SELECT open_price, high_price, low_price, close_price, price_change_rate,
               volume, individual, fore, institution, program,
               shrts_qty, ovr_shrts_qty, trde_wght, shrts_trde_price, shrts_avg_pric,
               date
            FROM daily_stock
            WHERE stock_code = %s         and date < %s
            ORDER BY date DESC
            LIMIT 30
        ) AS recent
        ORDER BY date ASC; 
        """,
        (stockCode, date)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    x_data = [row[0:15] for row in rows]
    res = inference(x_data, stockCode)
    return res

@app.route("/train/<stockCode>", methods=["POST"])
def train_stock(stockCode):
    print("요청받은 stockCode:", stockCode)

    if len(stockCode) != 6 or not stockCode.isdigit():
        return jsonify({"error": "Invalid stockCode"}), 400

    # 이미 학습된 .h5 파일이 있으면 실행하지 않음
    model_path = f"{stockCode}.h5"
    if os.path.exists(model_path):
        return jsonify({"status": "skipped", "message": "이미 학습된 모델이 있습니다."}), 200

    # 백그라운드 스레드 시작
    thread = Thread(target=fetch_and_run, args=(stockCode,))
    thread.start()

    return jsonify({"status": "OK", "message": f"{stockCode} 학습 시작됨"}), 202

@app.route("/inference/<stockCode>/<date>", methods=["GET"])
def inference_stock(stockCode, date):
    if len(stockCode) != 6 or not stockCode.isdigit():
        return jsonify({"error": "Invalid stockCode"}), 400

    # 백그라운드 스레드 시작
    res = inference_run(stockCode, date)

    return res

if __name__ == "__main__":
    # Thread(target=fetch_and_run, args=("005930",), daemon=True).start()
    app.run(host="127.0.0.1", port=5000, use_reloader=False)

