import streamlit as st
import numpy as np
import cv2
import json

# デバッグ情報を格納するための辞書
debug_info = {}

# --- ユーティリティ関数（変更なし） ---

# 画像をリサイズし、アスペクト比を維持
def resize_image(image, max_width=800):
    (h, w) = image.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        dim = (max_width, int(h * ratio))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized
    return image

# マニュアル座標入力用のUIを表示
def display_manual_input(image):
    st.subheader("手動座標入力")
    st.warning("自動補正に失敗しました。基準となる紙の4つの角を以下の順序で入力してください。")
    st.info("画像の左上、右上、右下、左下の順に入力してください。")

    # セッションステートで座標を保持
    if 'manual_coords' not in st.session_state:
        st.session_state.manual_coords = [None, None, None, None]

    cols = st.columns(4)
    labels = ["左上", "右上", "右下", "左下"]
    
    # ユーザーが座標を入力するための数値入力フィールド
    for i, label in enumerate(labels):
        with cols[i]:
            # X座標
            st.session_state.manual_coords[i] = st.number_input(
                f"{label} X座標 (0-{image.shape[1]})",
                min_value=0,
                max_value=image.shape[1],
                value=st.session_state.manual_coords[i][0] if st.session_state.manual_coords[i] else 0,
                key=f'x_coord_{i}'
            )
            # Y座標
            st.session_state.manual_coords[i] = (
                st.session_state.manual_coords[i],
                st.number_input(
                    f"{label} Y座標 (0-{image.shape[0]})",
                    min_value=0,
                    max_value=image.shape[0],
                    value=st.session_state.manual_coords[i][1] if st.session_state.manual_coords[i] and isinstance(st.session_state.manual_coords[i], tuple) else 0,
                    key=f'y_coord_{i}'
                )
            )

    # 4点全て入力されたかチェック
    if all(isinstance(coord, tuple) and len(coord) == 2 for coord in st.session_state.manual_coords):
        # 4点をNumpy配列に変換
        manual_points = np.array([
            st.session_state.manual_coords[0], st.session_state.manual_coords[1],
            st.session_state.manual_coords[2], st.session_state.manual_coords[3]
        ], dtype="float32")
        
        # 補正開始ボタンの表示
        if st.button("手動補正を開始"):
            return manual_points
    
    return None

# 画像のパースペクティブ変換（変更なし）
def four_point_transform(image, pts, target_width, target_height):
    rect = np.array([
        [0, 0], [target_width - 1, 0],
        [target_width - 1, target_height - 1], [0, target_height - 1]
    ], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(pts, rect)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    return warped, M

# パースペクティブ補正された画像から服のバウンディングボックスを抽出（変更なし）
def find_clothing_bounding_box(warped_image):
    # HSVに変換し、服の色範囲を検出
    hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
    
    # ここでは、青色の服を検出するための一般的な範囲を使用します
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # マスクを作成
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # モルフォロジー変換でノイズを除去し、領域を結合
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 輪郭を検出
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # 最大の輪郭を見つける（通常、それが服全体）
    largest_contour = max(contours, key=cv2.contourArea)
    
    # バウンディングボックスを取得
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # デバッグ情報としてマスクを追加
    debug_info['binary_mask'] = mask
    
    return (x, y, w, h), largest_contour

# 測定ロジック（変更なし）
def measure_clothing(bounding_box, paper_size_mm, pixels_per_metric):
    if bounding_box is None:
        return None
    
    x, y, w, h = bounding_box
    
    # 実際の服の寸法を計算
    # 着丈 (y軸方向)
    height_cm = h / pixels_per_metric 
    # 身幅 (x軸方向)
    width_cm = w / pixels_per_metric
    
    return height_cm, width_cm

# 4つの角を自動検出する（修正なし）
def find_quadrilateral(image):
    # 画像をグレースケールに変換し、ガウシアンブラーを適用
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # エッジ検出
    # Canny検出器を使用する前に、OpenCVが最も成功しやすいように画像を調整する
    # ユーザーのPhotoshopでの調整により、この部分の検出精度が大きく左右される
    edged = cv2.Canny(blurred, 50, 200)

    # 輪郭を見つける
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 輪郭を面積でソートし、最も大きなものを選択
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    # 4つの頂点を持つ四角形を見つける
    for c in contours:
        # 輪郭の周囲長を計算
        peri = cv2.arcLength(c, True)
        # 輪郭を近似し、頂点数を確認
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # 4つの頂点があり、閉じた形状であれば、それが基準の紙と仮定
        if len(approx) == 4:
            # 頂点の順序を (左上, 右上, 右下, 左下) に並べ替える
            points = approx.reshape(4, 2)
            
            # 頂点を正しい順序に並べ替えるヘルパー関数
            def order_points(pts):
                # 4点を初期化
                rect = np.zeros((4, 2), dtype = "float32")

                # 左上 (最小の合計) と右下 (最大の合計)
                s = pts.sum(axis = 1)
                rect[0] = pts[np.argmin(s)] # 左上
                rect[2] = pts[np.argmax(s)] # 右下

                # 右上 (最小の差) と左下 (最大の差)
                diff = np.diff(pts, axis = 1)
                rect[1] = pts[np.argmin(diff)] # 右上
                rect[3] = pts[np.argmax(diff)] # 左下

                return rect
            
            return order_points(points)

    return None

# 採寸プロセス全体を制御する関数
def process_measurement(image):
    
    # 基準となる紙のサイズ（縦51cm、横38cm - ユーザーのカスタムサイズ）
    PAPER_HEIGHT_CM = 51.0
    PAPER_WIDTH_CM = 38.0
    
    # OpenCVが処理しやすいように画像をリサイズ（ユーザーには見えない）
    processed_image = resize_image(image)

    # 1. 基準となる紙の4つの角を検出
    st.info("ステップ 1/3: 基準となる紙の4つの角を自動検出中...")
    
    # --- 検出に失敗した場合の処理を追加 ---
    quad = find_quadrilateral(processed_image)

    if quad is None:
        st.error("補正中にエラーが発生しました。基準となる紙（4つの角を持つ物体）を検出できませんでした。")
        st.warning("手動で座標を入力するか、画像（特に境界線）をさらに明確にして再試行してください。")
        
        # 手動入力UIを表示し、結果を取得
        manual_points = display_manual_input(processed_image)

        if manual_points is None:
            return # 手動入力が完了していない場合は終了

        quad = manual_points # 手動入力された座標を使用

    # 2. パースペクティブ変換を実行
    st.info("ステップ 2/3: パースペクティブ補正を実行中...")
    
    # 補正後のターゲットサイズをピクセルで定義（アスペクト比を維持）
    # ターゲットの幅と高さを紙の比率に合わせる
    # 服の採寸には縦長の比率で十分なため、ここでは縦長に調整
    TARGET_WIDTH = 800
    TARGET_HEIGHT = int(TARGET_WIDTH * (PAPER_HEIGHT_CM / PAPER_WIDTH_CM))
    
    # 補正後の画像を取得
    warped_image_bgr, M = four_point_transform(processed_image, quad, TARGET_WIDTH, TARGET_HEIGHT)

    # 3. 採寸と結果の表示
    st.info("ステップ 3/3: 服の寸法を測定中...")
    
    # ピクセルあたりのセンチメートル数を計算 (例: 800px / 38cm)
    pixels_per_cm = TARGET_WIDTH / PAPER_WIDTH_CM
    
    # 服のバウンディングボックスを見つける
    bounding_box, largest_contour = find_clothing_bounding_box(warped_image_bgr)
    
    # 測定結果を取得
    measurement_results = measure_clothing(bounding_box, (PAPER_HEIGHT_CM, PAPER_WIDTH_CM), pixels_per_cm)

    # 測定結果の表示
    st.success("採寸が完了しました！")
    
    if measurement_results:
        height_cm, width_cm = measurement_results
        
        st.subheader("📐 計測結果 (カスタム基準)")
        st.markdown(f"**着丈 (推定):** {height_cm:.1f} cm")
        st.markdown(f"**身幅 (推定):** {width_cm:.1f} cm")
        
        st.info("計測は服の外枠 (バウンディングボックス) に基づいています。")

    else:
        st.error("服の検出に失敗しました。服が背景（紙）と同じ色ではないか確認してください。")
        return

    # --- デバッグ情報（結果の視覚化）---
    
    # 補正された画像（デバッグ用）
    warped_image_display = warped_image_bgr.copy()
    
    # 服のバウンディングボックスを描画
    if bounding_box:
        x, y, w, h = bounding_box
        # 服の外枠を赤色で表示
        cv2.rectangle(warped_image_display, (x, y), (x + w, y + h), (0, 0, 255), 5)
        
    st.subheader("🐛 デバッグ情報")
    st.image(warped_image_display, channels="BGR", caption="パースペクティブ補正後の画像と推定バウンディングボックス")
    
    # 輪郭マスクの表示
    if 'binary_mask' in debug_info:
        st.image(debug_info['binary_mask'], caption="閾値処理後の画像（服が白く表示されているか確認）", use_column_width=True)
        st.markdown(f"Pixels Per Metric (1cmあたり): {pixels_per_cm:.2f} pixels")
        
    st.markdown(f"※このアプリは、縦{PAPER_HEIGHT_CM}cm、横{PAPER_WIDTH_CM}cmの紙の既知の寸法を基準としています。")


# --- Streamlit UI（変更なし） ---

st.title("👕 服の自動採寸アプリ (カスタム基準)")
st.markdown(f"服を縦51cm、横38cmの紙に置いて撮影した画像をアップロードしてください。")

uploaded_file = st.file_uploader("ファイルをここにドラッグアンドドロップしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ファイルの読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(image, channels="BGR", caption="アップロードされた画像")

    if st.button("採寸開始"):
        try:
            process_measurement(image)
        except Exception as e:
            st.error(f"計測中にエラーが発生しました。コードを確認してください: {e}")
