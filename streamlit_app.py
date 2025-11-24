import streamlit as st
from PIL import Image
import numpy as np
import cv2
from scipy.spatial import distance as dist
import utils  # utils.pyからorder_points関数をインポート

# =======================================================
# 📏 【カスタム基準寸法】 (縦 51cm, 横 38cm)
# =======================================================
KNOWN_WIDTH_CM = 38.0  # 既知の横幅 (短辺)
KNOWN_LENGTH_CM = 51.0 # 既知の縦幅 (長辺)

# ユーザーがクリックした座標を保存するためのセッションステート
if 'clicks' not in st.session_state:
    st.session_state.clicks = []
if 'img_data' not in st.session_state:
    st.session_state.img_data = None # パースペクティブ補正後の画像データ (RGB形式)
if 'ppm' not in st.session_state:
    st.session_state.ppm = None # Pixels Per Metric

# =======================================================
# 📏 【自動基準検出＆補正ロジック関数】
# =======================================================

def process_image_and_get_ppm(image_np, known_width, known_length):
    """
    画像を前処理し、パースペクティブ補正を行い、Pixels Per Metricを計算して返す
    """
    # 1. 画像の前処理
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 100)
    
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # 2. 輪郭の検出
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 3. 基準紙（四角形）の特定
    paper_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            paper_contour = approx
            break
            
    if paper_contour is None:
        raise Exception("基準となる紙（4つの角を持つ物体）を検出できませんでした。撮影環境を確認してください。")

    # 4. パースペクティブ補正のための処理
    pts = paper_contour.reshape(4, 2)
    rect = utils.order_points(pts) # utils.pyの関数を使用

    # 補正後の画像の理想的なサイズを決定 (縦51cm:横38cmの比率を維持)
    ratio_custom = known_length / known_width
    W_ideal = 1000  # 補正後の画像幅の仮設定（ピクセル数）
    H_ideal = int(W_ideal * ratio_custom)

    # 5. ワープ変換（パースペクティブ補正）
    dst = np.array([
        [0, 0],
        [W_ideal - 1, 0],
        [W_ideal - 1, H_ideal - 1],
        [0, H_ideal - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_np, M, (W_ideal, H_ideal))
    
    # 6. Pixels Per Metric の計算
    pixels_per_metric = W_ideal / known_width 

    # BGRをRGBに変換して保存 (streamlitでの表示用)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    return warped_rgb, pixels_per_metric

# =======================================================
# 💡 【手動計測ロジック関数】
# =======================================================

def calculate_measurements(clicks, ppm):
    """
    ユーザーのクリック座標から着丈と身幅を計算する
    """
    results = {}
    
    # 着丈 (2点: 始点, 終点) の計測
    if len(clicks) >= 2:
        p1 = clicks[0]
        p2 = clicks[1]
        # ピクセル差の絶対値を取得（縦方向の距離）
        length_pixels = abs(p1['y'] - p2['y'])
        length_cm = length_pixels / ppm
        results["**着丈 (縦の距離)**"] = length_cm
        
    # 身幅 (2点: 始点, 終点) の計測
    if len(clicks) >= 4:
        p3 = clicks[2]
        p4 = clicks[3]
        # ピクセル差の絶対値を取得（横方向の距離）
        width_pixels = abs(p3['x'] - p4['x'])
        width_cm = width_pixels / ppm
        results["**身幅 (横の距離)**"] = width_cm
        
    return results

# =======================================================
# 📱 Streamlit UI 部分
# =======================================================

st.title('👕 服の自動採寸アプリ (手動クリック指定)')
st.subheader('服を縦51cm、横38cmの紙に置いて撮影した画像をアップロードしてください。')
st.info('**手順：** 1. 画像アップロード -> 2. 「補正開始」 -> 3. 補正後の画像で**座標を手動で4点入力** -> 4. 「採寸実行」')

# 1. ファイルアップロード
uploaded_file = st.file_uploader("採寸したい服の画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 画像をPIL/Numpyでロード
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB')) 
    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    st.image(image, caption='アップロードされた画像', use_column_width=True)

    # 2. パースペクティブ補正の実行
    if st.button('1. 補正開始 (自動パースペクティブ補正)'):
        with st.spinner('補正中...'):
            try:
                # 補正後の画像とPPMを取得
                warped_rgb, pixels_per_metric = process_image_and_get_ppm(
                    image_np_bgr, KNOWN_WIDTH_CM, KNOWN_LENGTH_CM
                )
                # セッションステートに保存
                st.session_state.img_data = warped_rgb
                st.session_state.ppm = pixels_per_metric
                st.session_state.clicks = [] # 補正が完了したらクリックをリセット
                st.success('補正が完了しました。下の画像で座標を確認し、計測点を入力してください。')
            except Exception as e:
                st.session_state.img_data = None
                st.session_state.ppm = None
                st.error(f"補正中にエラーが発生しました: {e}")

# 3. 補正後画像の表示と座標取得（手動入力でシミュレーション）
if st.session_state.img_data is not None:
    st.markdown("### 2. 計測点 手動入力")
    st.warning('**入力の順番を守ってください:** 1, 2点目: 着丈の始点と終点 (縦方向) / 3, 4点目: 身幅の始点と終点 (横方向)')

    # 補正済み画像を表示
    st.image(st.session_state.img_data, caption="補正済みの画像", use_column_width=True)
    st.caption("この画像内のピクセル座標を基に、以下のフィールドに入力してください。")

    # 既存のクリック数を表示
    num_clicks = len(st.session_state.clicks)
    
    # 現在設定されている点を表示
    point_names = ["着丈始点 (P1)", "着丈終点 (P2)", "身幅始点 (P3)", "身幅終点 (P4)"]
    st.markdown("#### 📍 現在の指定点")
    for i in range(4):
        if i < num_clicks:
            point = st.session_state.clicks[i]
            st.write(f"**{point_names[i]}:** X={point['x']}, Y={point['y']}")
        else:
            st.write(f"**{point_names[i]}:** <未設定>")
            
    # 新しいクリック点を追加するUI
    if num_clicks < 4:
        st.markdown("---")
        st.markdown(f"#### 💾 {point_names[num_clicks]} の座標を入力 (残り {4 - num_clicks} 点)")
        
        # 画面幅に合わせた入力
        col_x, col_y = st.columns(2)
        # 補正後画像サイズ W_ideal=1000, H_ideal=int(1000 * 51/38) = 1342
        max_x = 1000
        max_y = int(1000 * KNOWN_LENGTH_CM / KNOWN_WIDTH_CM) # ~1342
        
        # value=0で初期値を設定
        new_x = col_x.number_input("X座標 (Pixels):", min_value=0, max_value=max_x, key='new_x', step=1, value=0)
        new_y = col_y.number_input("Y座標 (Pixels):", min_value=0, max_value=max_y, key='new_y', step=1, value=0)
        
        if st.button('点を追加して保存'):
            st.session_state.clicks.append({'x': new_x, 'y': new_y})
            st.experimental_rerun()
    
    st.markdown("---")
    if st.button('全ての指定点をリセット'):
        st.session_state.clicks = []
        st.experimental_rerun()
        
    # 4. 採寸の実行
    if num_clicks >= 4:
        if st.button('3. 採寸実行'):
            with st.spinner('計算中...'):
                try:
                    # 計測ロジックを呼び出す
                    measurements = calculate_measurements(st.session_state.clicks, st.session_state.ppm)
                    
                    # 計測成功時の表示ロジック
                    st.success('採寸が完了しました！')
                    st.markdown("### 📐 計測結果 (手動指定)")

                    for key, value in measurements.items():
                        st.write(f"* **{key}:** {value:.1f} cm")
                    
                    st.info("着丈は点1(P1)と点2(P2)の縦の距離、身幅は点3(P3)と点4(P4)の横の距離として計算されています。")
                    
                except Exception as e: 
                    st.error(f"計測中にエラーが発生しました: {e}")
            
# 最後の注意書き
st.markdown("---")
st.info('※このアプリは、縦51cm、横38cmの紙の既知の寸法を基準としています。')
