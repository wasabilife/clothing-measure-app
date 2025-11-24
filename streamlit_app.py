# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2 
from scipy.spatial import distance as dist
import utils  # ← utils.pyから関数をインポート

# A3用紙の既知の寸法（例: 短辺）をセンチメートルで定義
# 基準オブジェクトの既知の長さとして利用します
KNOWN_WIDTH_CM = 29.7 

# =======================================================
# 📏 【採寸ロジック関数】 A3用紙を基準に計算する
# =======================================================

def measure_clothing(image_np, known_width):
    """
    A3用紙を検出し、パースペクティブ補正を行い、Pixels Per Metricを計算する
    """
    
    # 1. 画像の前処理
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)
    
    # エッジの閉処理 (輪郭の途切れを埋める)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # 2. 輪郭の検出
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 輪郭の面積が大きい順にソート（A3用紙が最大面積である可能性が高いと仮定）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 3. A3用紙（四角形）の特定と4点の抽出
    paper_contour = None
    for c in contours:
        # 周囲の長さから、輪郭の近似多角形を取得
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 4つの頂点を持つ輪郭をA3用紙として採用
        if len(approx) == 4:
            paper_contour = approx
            break
            
    if paper_contour is None:
        raise Exception("A3画用紙（4つの角を持つ物体）を検出できませんでした。撮影環境を確認してください。")

    # 4. パースペクティブ補正のための処理
    # 検出した4つの角を utils.py の関数で順序付け (左上、右上、右下、左下の順)
    pts = paper_contour.reshape(4, 2)
    rect = utils.order_points(pts) 
    (tl, tr, br, bl) = rect # 座標を変数に展開

    # 補正後の画像の理想的なサイズを決定（A3の比率 420:297 を維持）
    # A3の短辺29.7cmを基準に、420/297の比率で長辺を計算
    ratio_a3 = 420.0 / 297.0 
    W_ideal = 1000  # 補正後の画像幅の仮設定（任意のピクセル数）
    H_ideal = int(W_ideal * ratio_a3)

    # 5. ワープ変換（パースペクティブ補正）
    # 補正後のターゲット座標 (理想的な長方形)
    dst = np.array([
        [0, 0],
        [W_ideal - 1, 0],
        [W_ideal - 1, H_ideal - 1],
        [0, H_ideal - 1]], dtype="float32")

    # 変換行列を取得し、画像をワープ変換
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_np, M, (W_ideal, H_ideal))
    
    # 6. Pixels Per Metric の計算
    # 補正後の短辺のピクセル数 (W_ideal) と実際の長さ (KNOWN_WIDTH_CM = 29.7cm) から計算
    pixels_per_metric = W_ideal / known_width 
    
   # =======================================================
    # 7. 服の寸法計測（バウンディングボックスによる簡易計測）
    # =======================================================
    
    # 補正後の画像をグレースケールにし、服を際立たせる
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # 閾値処理：A3の白と服の黒を分離する (ここではしきい値100を使用。色によって調整が必要)
    # THRESH_BINARY_INV で、服の部分が白 (255) になるように反転させる
    _, thresh = cv2.threshold(warped_gray, 100, 255, cv2.THRESH_BINARY_INV) 

    # 再度輪郭を検出
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        # 服の輪郭が見つからない場合のエラー
        raise Exception("補正後の画像から服の輪郭を検出できませんでした。服が白すぎるか、A3用紙とのコントラストが不足しています。")

    # 最大の輪郭（服）を抽出
    c = max(cnts, key=cv2.contourArea)

    # 最小外接矩形を取得 (x, y, 幅w, 高さh をピクセルで取得)
    x, y, w_pixels, h_pixels = cv2.boundingRect(c)

    # Pixels Per Metric を使ってCMに変換
    width_cm = w_pixels / pixels_per_metric
    length_cm = h_pixels / pixels_per_metric

    # 輪郭の検出と計測ロジックは残し、ここでは処理後の画像と Pixels Per Metric を返す
    return {
        "**着丈 (推定)**": length_cm,
        "**身幅 (推定)**": width_cm,
        "備考": "計測は服の外枠（バウンディングボックス）に基づいています。",
        "debug_image": thresh,  # ← デバッグ用の閾値画像を辞書に追加
        "pixels_per_metric": pixels_per_metric # ← デバッグ用の値も追加
    }
    
# =======================================================
# 📱 Streamlit UI 部分
# =======================================================

st.title('👕 服の自動採寸アプリ (A3基準)')
st.subheader('服をA3画用紙に置いて撮影した画像をアップロードしてください。')

# ユーザーからのファイルアップロードを許可
uploaded_file = st.file_uploader("採寸したい服の画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_column_width=True)
    
    # PIL ImageをOpenCVが扱えるNumpy配列に変換（BGR形式に変換）
    image_np = np.array(image.convert('RGB')) 
    # OpenCVはBGR形式を使うため、RGBをBGRに変換する処理を開発時に追加する必要があります
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
    
    # 採寸ボタン
if st.button('採寸開始'):
    # 処理状況を通知
    with st.spinner('計測中...画像解析と計算を行っています。'):
        
        try:
            # 採寸ロジックを呼び出す
            measurements = measure_clothing(image_np, KNOWN_WIDTH_CM)
            
            # 計測成功時の表示ロジック
            st.success('採寸が完了しました！')
            st.markdown("### 📐 計測結果 (A3基準)")

            # 結果表示ループ
            remarks = measurements.get("備考", None)
            
            for key, value in measurements.items():
                if key == "備考":
                    continue
                st.write(f"* **{key}:** {value:.1f} cm")
            
            if remarks:
                st.info(remarks)
        
        # 🚨 ここで try ブロックを閉じる except を追加 🚨
        except Exception as e: 
            # tryブロック内でエラーが発生したら、ここでキャッチする
            st.error(f"計測中にエラーが発生しました。コードを確認してください: {e}")
            
        # 🚨 デバッグ表示は try/except の「後」に続ける 🚨
        # try/except と同じインデントレベル (レベル 3) に戻す
        debug_img = measurements.get("debug_image", None)
        if debug_img is not None:
            st.header("🐛 デバッグ情報")
            st.image(debug_img, caption="閾値処理後の画像（服が白く表示されているか確認）", use_column_width=True)
            st.write(f"Pixels Per Metric (1cmあたり): {measurements.get('pixels_per_metric', 'N/A'):.2f} pixels")
            
# st.info(...) は if ブロックの外側 (レベル 1) にある
st.info('※このアプリは、A3画用紙の既知の寸法を基準としています。')
