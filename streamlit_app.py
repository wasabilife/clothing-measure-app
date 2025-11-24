import streamlit as st
from PIL import Image
import numpy as np
import cv2
from scipy.spatial import distance as dist
# utils.pyからのインポートが前提です。別途このファイルがあることを確認してください。
import utils  

# =======================================================
# 📏 【カスタム基準寸法】 (縦 51cm, 横 38cm)
# =======================================================
# 基準オブジェクトの既知の寸法をセンチメートルで定義
KNOWN_WIDTH_CM = 38.0  # 既知の横幅 (短辺)
KNOWN_LENGTH_CM = 51.0 # 既知の縦幅 (長辺)

# =======================================================
# 📏 【採寸ロジック関数】 カスタム紙を基準に計算する
# 引数は3つ (image_np, known_width, known_length) です。
# =======================================================

def measure_clothing(image_np, known_width, known_length):
    """
    カスタムサイズの紙を検出し、パースペクティブ補正を行い、Pixels Per Metricを計算する
    """
    
    # 1. 画像の前処理
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Cannyエッジ検出
    edged = cv2.Canny(blurred, 50, 100)
    
    # エッジの閉処理 (輪郭の途切れを埋める)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # 2. 輪郭の検出
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 輪郭の面積が大きい順にソート（基準紙が最大面積である可能性が高いと仮定）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 3. 基準紙（四角形）の特定と4点の抽出
    paper_contour = None
    for c in contours:
        # 周囲の長さから、輪郭の近似多角形を取得
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 4つの頂点を持つ輪郭を基準紙として採用
        if len(approx) == 4:
            paper_contour = approx
            break
            
    if paper_contour is None:
        raise Exception("基準となる紙（4つの角を持つ物体）を検出できませんでした。撮影環境を確認してください。")

    # 4. パースペクティブ補正のための処理
    pts = paper_contour.reshape(4, 2)
    # 検出した4つの角を utils.py の関数で順序付け
    rect = utils.order_points(pts) 
    (tl, tr, br, bl) = rect

    # 補正後の画像の理想的なサイズを決定 (縦51cm:横38cmの比率を維持)
    ratio_custom = known_length / known_width
    W_ideal = 1000  # 補正後の画像幅の仮設定（ピクセル数）
    H_ideal = int(W_ideal * ratio_custom)

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
    # 補正後の短辺のピクセル数 (W_ideal) と実際の長さ (KNOWN_WIDTH_CM = 38.0cm) から計算
    pixels_per_metric = W_ideal / known_width 
    
    # =======================================================
    # 7. 服の寸法計測（バウンディングボックスによる簡易計測）
    # =======================================================
    
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # 閾値処理: 黒い紙の上に明るい色の服を置いていることを想定し、THRESH_BINARYを使用
    # 明るいピクセル（服）を白 (255) に、暗いピクセル（背景）を黒 (0) にする
    # 閾値 100 を使用
    _, thresh = cv2.threshold(warped_gray, 100, 255, cv2.THRESH_BINARY) 

    # 再度輪郭を検出
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        # 服の輪郭が見つからない場合のエラー
        raise Exception("補正後の画像から服の輪郭を検出できませんでした。紙と服のコントラストを確認してください。")

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
        "debug_image": thresh,  # デバッグ用の閾値画像を辞書に追加
        "pixels_per_metric": pixels_per_metric # デバッグ用の値も追加
    }
    
# =======================================================
# 📱 Streamlit UI 部分
# =======================================================

st.title('👕 服の自動採寸アプリ (カスタム基準)')
st.subheader('服を縦51cm、横38cmの紙に置いて撮影した画像をアップロードしてください。')

# ユーザーからのファイルアップロードを許可
uploaded_file = st.file_uploader("採寸したい服の画像をアップロード", type=['jpg', 'jpeg', 'png'])

image_np = None # 初期化
if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_column_width=True)
    
    # PIL ImageをOpenCVが扱えるNumpy配列に変換（BGR形式に変換）
    image_np = np.array(image.convert('RGB')) 
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
    
# 採寸ボタン
if st.button('採寸開始'):
    if uploaded_file is None:
        st.warning("画像をアップロードしてから採寸を開始してください。")
    elif image_np is None:
        st.error("画像の読み込みに失敗しました。")
    else:
        # 処理状況を通知
        with st.spinner('計測中...画像解析と計算を行っています。'):
            
            measurements = {} # measurements を try ブロックの外で初期化
            try:
                # 採寸ロジックを呼び出す
                # 引数3つで呼び出し (以前のエラー修正済み)
                measurements = measure_clothing(image_np, KNOWN_WIDTH_CM, KNOWN_LENGTH_CM)
                
                # 計測成功時の表示ロジック
                st.success('採寸が完了しました！')
                st.markdown("### 📐 計測結果 (カスタム基準)")

                # 結果表示ループ
                remarks = measurements.get("備考", None)
                
                for key, value in measurements.items():
                    # デバッグ情報と備考は結果表示から除外
                    if key == "備考" or key == "debug_image" or key == "pixels_per_metric":
                        continue
                    # 数値のみを .1f でフォーマット
                    st.write(f"* **{key}:** {value:.1f} cm")
                
                if remarks:
                    st.info(remarks)
            
            # tryブロック内でエラーが発生したら、ここでキャッチする
            except Exception as e: 
                st.error(f"計測中にエラーが発生しました。コードを確認してください: {e}")
                
            # デバッグ表示
            debug_img = measurements.get("debug_image", None)
            debug_ppm = measurements.get("pixels_per_metric", 'N/A')

            if debug_img is not None:
                st.header("🐛 デバッグ情報")
                # 閾値画像をそのまま表示
                st.image(debug_img, caption="閾値処理後の画像（服が白く表示されているか確認）", use_column_width=True)
                
                # Pixels Per Metric の表示
                if isinstance(debug_ppm, float):
                    st.write(f"Pixels Per Metric (1cmあたり): {debug_ppm:.2f} pixels")
                else:
                    st.write(f"Pixels Per Metric (1cmあたり): {debug_ppm}")
            
# st.info(...) は if ブロックの外側にある
st.info('※このアプリは、縦51cm、横38cmの紙の既知の寸法を基準としています。コントラストを上げるため、服とは逆の色の紙（例：黒い紙）の使用を推奨します。')
