import streamlit as st
import numpy as np
import cv2
import json

# デバッグ情報を格納するための辞書
debug_info = {}

# --- ユーティリティ関数 ---

# 画像をリサイズし、アスペクト比を維持
def resize_image(image, max_width=800):
    (h, w) = image.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        dim = (max_width, int(h * ratio))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized
    return image

# 画像のパースペクティブ変換
def four_point_transform(image, pts, target_width, target_height):
    rect = np.array([
        [0, 0], [target_width - 1, 0],
        [target_width - 1, target_height - 1], [0, target_height - 1]
    ], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(pts, rect)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    return warped, M

# 測定ロジック
def measure_clothing(measurement_points, target_width, target_height, paper_width_cm, paper_height_cm):
    
    # 補正後の画像サイズ（ピクセル）
    x_px = measurement_points[:, 0]
    y_px = measurement_points[:, 1]
    
    # ピクセルあたりのセンチメートル数を計算
    pixels_per_cm_x = target_width / paper_width_cm
    pixels_per_cm_y = target_height / paper_height_cm
    
    # 着丈 (Y軸の最大値と最小値の差)
    # ユーザーが指定した着丈の始まり(0)と終わり(1)
    height_px = abs(y_px[1] - y_px[0])
    
    # 身幅 (X軸の最大値と最小値の差)
    # ユーザーが指定した身幅の始まり(2)と終わり(3)
    width_px = abs(x_px[3] - x_px[2])
    
    # 実際の服の寸法を計算
    # 縦方向の計測にはyのスケール、横方向の計測にはxのスケールを使用
    height_cm = height_px / pixels_per_cm_y
    width_cm = width_px / pixels_per_cm_x
    
    return height_cm, width_cm, pixels_per_cm_x, pixels_per_cm_y

# --- Streamlit UIと状態管理 ---

# 基準となる紙のサイズ（縦51cm、横38cm - ユーザーのカスタムサイズ）
PAPER_HEIGHT_CM = 51.0
PAPER_WIDTH_CM = 38.0

# 補正後のターゲットサイズをピクセルで定義（アスペクト比を維持）
TARGET_WIDTH = 800
TARGET_HEIGHT = int(TARGET_WIDTH * (PAPER_HEIGHT_CM / PAPER_WIDTH_CM))

def init_session_state():
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'paper_coords' not in st.session_state:
        st.session_state.paper_coords = [None] * 4 # [左上, 右上, 右下, 左下]
    if 'measure_coords' not in st.session_state:
        st.session_state.measure_coords = [None] * 4 # [着丈上, 着丈下, 身幅左, 身幅右]

def main():
    init_session_state()

    st.title("📐 服のカスタム採寸アプリ")
    st.markdown("---")
    
    # ファイルアップローダー
    uploaded_file = st.file_uploader("ステップ 0: 画像をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if st.session_state.original_image is None or st.session_state.uploaded_file_name != uploaded_file.name:
            # 新しいファイルがアップロードされた場合、状態をリセット
            st.session_state.step = 1
            st.session_state.uploaded_file_name = uploaded_file.name
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.original_image = resize_image(original_image_bgr) # 処理用にリサイズ

    if st.session_state.original_image is None:
        st.info("縦51cm、横38cmの紙の上に服を置いて撮影した画像をアップロードしてください。")
        return

    # --- ステップ 1: 紙の角の指定とパースペクティブ補正 ---
    if st.session_state.step == 1:
        st.header("ステップ 1: 基準となる紙の4つの角を入力")
        st.warning("紙の角の正確な**X, Y座標**を入力してください。座標はPhotoshopなどのツールで確認できます。")
        st.info(f"アップロードされた画像サイズ: {st.session_state.original_image.shape[1]} x {st.session_state.original_image.shape[0]} (リサイズ後)")
        
        st.image(st.session_state.original_image, channels="BGR", caption="元の画像（この画像を参考に座標を入力）", use_column_width=True)

        labels = ["左上", "右上", "右下", "左下"]
        col_x, col_y = st.columns(2)
        
        for i, label in enumerate(labels):
            with col_x if i % 2 == 0 else col_y:
                # ユーザーが座標を入力するための数値入力フィールド
                st.subheader(f"{label}の座標")
                x_val = st.number_input(
                    f"{label} X座標",
                    min_value=0,
                    max_value=st.session_state.original_image.shape[1],
                    value=st.session_state.paper_coords[i][0] if st.session_state.paper_coords[i] else 0,
                    key=f'paper_x_{i}'
                )
                y_val = st.number_input(
                    f"{label} Y座標",
                    min_value=0,
                    max_value=st.session_state.original_image.shape[0],
                    value=st.session_state.paper_coords[i][1] if st.session_state.paper_coords[i] else 0,
                    key=f'paper_y_{i}'
                )
                st.session_state.paper_coords[i] = (x_val, y_val)


        if st.button("画像を補正し、ステップ2へ進む", key="go_to_step2"):
            try:
                # 4点をNumpy配列に変換
                paper_points = np.array(st.session_state.paper_coords, dtype="float32")
                
                # 補正を実行
                warped_image_bgr, _ = four_point_transform(
                    st.session_state.original_image, paper_points, TARGET_WIDTH, TARGET_HEIGHT
                )
                
                st.session_state.processed_image = warped_image_bgr
                st.session_state.step = 2
                st.experimental_rerun() # ステップ2へ移行

            except Exception as e:
                st.error(f"パースペクティブ補正に失敗しました。座標入力が間違っている可能性があります: {e}")
                st.exception(e)

    # --- ステップ 2: 着丈・身幅の採寸点を指定 ---
    elif st.session_state.step == 2:
        st.header("ステップ 2: 着丈・身幅の計測点を入力")
        st.info("補正後の画像を見ながら、服の採寸に必要な4点の正確なX, Y座標を入力してください。")
        st.warning(f"補正後の画像サイズ: {TARGET_WIDTH} x {TARGET_HEIGHT}。座標はこの範囲内で入力してください。")
        
        # 補正後の画像を表示
        st.image(st.session_state.processed_image, channels="BGR", 
                 caption="補正後の画像（この画像を参考に座標を入力）", use_column_width=True)

        labels = ["着丈 (上端)", "着丈 (下端)", "身幅 (左端)", "身幅 (右端)"]
        col_x, col_y = st.columns(2)

        for i, label in enumerate(labels):
            with col_x if i % 2 == 0 else col_y:
                st.subheader(f"{label}の座標")
                x_val = st.number_input(
                    f"{label} X座標 (0-{TARGET_WIDTH})",
                    min_value=0,
                    max_value=TARGET_WIDTH,
                    value=st.session_state.measure_coords[i][0] if st.session_state.measure_coords[i] else 0,
                    key=f'measure_x_{i}'
                )
                y_val = st.number_input(
                    f"{label} Y座標 (0-{TARGET_HEIGHT})",
                    min_value=0,
                    max_value=TARGET_HEIGHT,
                    value=st.session_state.measure_coords[i][1] if st.session_state.measure_coords[i] else 0,
                    key=f'measure_y_{i}'
                )
                st.session_state.measure_coords[i] = (x_val, y_val)
        
        # 戻るボタン
        if st.button("← ステップ1に戻る", key="back_to_step1"):
            st.session_state.step = 1
            st.experimental_rerun()

        if st.button("採寸結果を表示", key="show_results"):
            st.session_state.step = 3
            st.experimental_rerun() # ステップ3へ移行

    # --- ステップ 3: 結果の表示 ---
    elif st.session_state.step == 3:
        st.header("ステップ 3: 採寸結果")
        
        # 測定ポイントをNumpy配列に変換
        measure_points = np.array(st.session_state.measure_coords, dtype="float32")
        
        try:
            height_cm, width_cm, pixels_per_cm_x, pixels_per_cm_y = measure_clothing(
                measure_points, TARGET_WIDTH, TARGET_HEIGHT, PAPER_WIDTH_CM, PAPER_HEIGHT_CM
            )

            st.success("採寸が完了しました！")
            
            st.subheader("📐 計測結果")
            st.markdown(f"**着丈 (推定):** **{height_cm:.1f} cm**")
            st.markdown(f"**身幅 (推定):** **{width_cm:.1f} cm**")
            
            st.info("結果は、カスタム基準（縦51cm、横38cm）と、手動で指定した4点に基づいて計算されています。")
            
            # --- 結果の視覚化（デバッグ情報として） ---
            warped_image_display = st.session_state.processed_image.copy()
            
            # 計測点を描画: [着丈上(青), 着丈下(青), 身幅左(緑), 身幅右(緑)]
            colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0)] # 青, 青, 緑, 緑
            labels_draw = ["着丈上", "着丈下", "身幅左", "身幅右"]

            for i, (x, y) in enumerate(st.session_state.measure_coords):
                x_int, y_int = int(x), int(y)
                cv2.circle(warped_image_display, (x_int, y_int), 10, colors[i], -1)
                cv2.putText(warped_image_display, labels_draw[i], (x_int + 15, y_int), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)

            # 着丈の線
            cv2.line(warped_image_display, 
                     (int(st.session_state.measure_coords[0][0]), int(st.session_state.measure_coords[0][1])), 
                     (int(st.session_state.measure_coords[1][0]), int(st.session_state.measure_coords[1][1])), 
                     (255, 0, 0), 3) # 青線
            
            # 身幅の線
            cv2.line(warped_image_display, 
                     (int(st.session_state.measure_coords[2][0]), int(st.session_state.measure_coords[2][1])), 
                     (int(st.session_state.measure_coords[3][0]), int(st.session_state.measure_coords[3][1])), 
                     (0, 255, 0), 3) # 緑線


            st.subheader("🐛 計測点と結果の確認")
            st.image(warped_image_display, channels="BGR", caption="計測点を描画した補正後の画像", use_column_width=True)
            
            st.markdown(f"**詳細スケール情報:**")
            st.markdown(f"・横方向 (1cmあたり): {pixels_per_cm_x:.2f} pixels")
            st.markdown(f"・縦方向 (1cmあたり): {pixels_per_cm_y:.2f} pixels")


        except Exception as e:
            st.error(f"計測の計算中にエラーが発生しました: {e}")
            st.exception(e)
            
        # 戻るボタン
        if st.button("← ステップ2に戻る", key="back_to_step2"):
            st.session_state.step = 2
            st.experimental_rerun()

# アプリケーションの実行
if __name__ == '__main__':
    main()
