import streamlit as st
from PIL import Image
import numpy as np
import cv2
from scipy.spatial import distance as dist
import utils  # utils.pyからorder_points関数をインポート

# =======================================================
# 📏 【カスタム基準寸法】 (縦 51cm, 横 38cm)
# =======================================================
KNOWN_WIDTH_CM = 38.0  # Known width in cm (shorter side)
KNOWN_LENGTH_CM = 51.0 # Known length in cm (longer side)

# Initialize Session State for measurements
if 'clicks' not in st.session_state:
    st.session_state.clicks = []
if 'img_data' not in st.session_state:
    st.session_state.img_data = None 
if 'ppm' not in st.session_state:
    st.session_state.ppm = None 

# =======================================================
# 📏 【自動基準検出＆補正ロジック関数】
# =======================================================

def process_image_and_get_ppm(image_np, known_width, known_length):
    """
    画像を前処理し、パースペクティブ補正を行い、Pixels Per Metricを計算して返す
    
    Args:
        image_np (np.array): RGB形式の画像Numpy配列
        known_width (float): 基準紙の既知の横幅 (cm)
        known_length (float): 基準紙の既知の縦幅 (cm)

    Returns:
        tuple: (補正後のRGB画像Numpy配列, Pixels Per Metric)
    """
    # Streamlitから来た画像は通常RGBなので、BGRに変換してOpenCVで処理
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        # カラー画像でない場合はエラー
        raise ValueError("画像がRGB形式ではありません。")

    # 1. 画像の前処理
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # エッジ検出の閾値を調整 (コントラストの低い画像に対応)
    edged = cv2.Canny(blurred, 30, 150)
    
    # 輪郭を明確にするための膨張・収縮処理
    kernel = np.ones((3,3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)

    # 2. 輪郭の検出
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 3. 基準紙（四角形）の特定
    paper_contour = None
    # 最小面積の閾値を設定 (画像サイズに応じて調整)
    min_area_threshold = image_np.shape[0] * image_np.shape[1] * 0.05 # 画像全体の5%以上の面積を持つこと
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # 4つの角を持ち、かつ一定以上の面積を持つものを基準紙として採用
        if len(approx) == 4 and cv2.contourArea(c) > min_area_threshold:
            paper_contour = approx
            break
            
    if paper_contour is None:
        # 検出失敗の場合は例外を投げる
        raise Exception("基準となる紙（4つの角を持つ物体）を検出できませんでした。コントラストを上げるか、明るい場所で撮影してください。")

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
    warped = cv2.warpPerspective(image_bgr, M, (W_ideal, H_ideal)) # BGR画像に対してワープ変換

    # 6. Pixels Per Metric の計算
    pixels_per_metric = W_ideal / known_width 

    # BGRをRGBに変換して返す (streamlitでの表示用)
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
    # 画像をPIL/Numpyでロード (RGB形式で読み込む)
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB')) 
    
    st.image(image, caption='アップロードされた画像', use_column_width=True)

    # 2. パースペクティブ補正の実行
    if st.button('1. 補正開始 (自動パースペクティブ補正)'):
        with st.spinner('補正中...'):
            try:
                # 補正後の画像とPPMを取得
                warped_rgb, pixels_per_metric = process_image_and_get_ppm(
                    image_np, KNOWN_WIDTH_CM, KNOWN_LENGTH_CM
                )
                # セッションステートに保存
                st.session_state.img_data = warped_rgb
                st.session_state.ppm = pixels_per_metric
                st.session_state.clicks = [] # 補正が完了したらクリックをリセット
                st.success('補正が完了しました。下の画像で座標を確認し、計測点を入力してください。')
            except Exception as e:
                st.session_state.img_data = None
                st.session_state.ppm = None
                # エラーメッセージを明確に表示
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
    # 最後の注意書き
st.markdown("---")
st.info('※このアプリは、縦51cm、横38cmの紙の既知の寸法を基準としています。')
