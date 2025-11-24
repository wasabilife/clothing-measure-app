import streamlit as st
import numpy as np
import cv2
import json

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
    
    # 着丈 (Y軸の最大値と最小値の差) - ユーザーが指定した着丈の始まり(0)と終わり(1)
    height_px = abs(y_px[1] - y_px[0])
    
    # 身幅 (X軸の最大値と最小値の差) - ユーザーが指定した身幅の始まり(2)と終わり(3)
    width_px = abs(x_px[3] - x_px[2])
    
    # 実際の服の寸法を計算
    height_cm = height_px / pixels_per_cm_y
    width_cm = width_px / pixels_per_cm_x
    
    return height_cm, width_cm, pixels_per_cm_x, pixels_per_cm_y

# --- Streamlit UIと状態管理 ---

# 基準となる紙のサイズ（縦51cm、横38cm - ユーザーのカスタムサイズ）
PAPER_HEIGHT_CM = 51.0
PAPER_WIDTH_CM = 38.0

# 補正後のターゲットサイズをピクセルで定義
TARGET_WIDTH = 800
TARGET_HEIGHT = int(TARGET_WIDTH * (PAPER_HEIGHT_CM / PAPER_WIDTH_CM))

def init_session_state():
    # アプリのステップ (1: 紙の角指定, 2: 採寸点指定, 3: 結果表示)
    if 'step' not in st.session_state:
        st.session_state.step = 1
    # 現在編集中のポイントのインデックス
    if 'active_point_index' not in st.session_state:
        st.session_state.active_point_index = 0
    
    # 画像データ
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
        
    # 座標データ [左上, 右上, 右下, 左下]
    if 'paper_coords' not in st.session_state:
        st.session_state.paper_coords = [None] * 4 
    # 採寸データ [着丈上, 着丈下, 身幅左, 身幅右]
    if 'measure_coords' not in st.session_state:
        st.session_state.measure_coords = [None] * 4 

# ポイントの入力を処理し、次のポイントへ進める
def handle_coordinate_input(coords_list_key, point_index, x_val, y_val, next_step_label):
    # 現在のポイントを保存
    st.session_state[coords_list_key][point_index] = (x_val, y_val)
    
    # 次のポイントへ進む、または次のステップへ移行
    if point_index < len(st.session_state[coords_list_key]) - 1:
        st.session_state.active_point_index += 1
    else:
        st.session_state.active_point_index = 0
        st.session_state.step += 1
    
    st.experimental_rerun()

# 座標入力UIの共通ロジック
def coordinate_input_ui(image, coords_list_key, labels, is_original_image):
    
    # 画像の表示（座標確認用）
    st.image(image, channels="BGR", caption=f"【現在編集中の画像】 サイズ: {image.shape[1]} x {image.shape[0]}", use_column_width=True)

    # 現在の座標リストのコピーを作成し、Noneを(0, 0)に初期化
    current_coords = [
        (0, 0) if coord is None else coord 
        for coord in st.session_state[coords_list_key]
    ]

    # --- 1. 選択中のポイントをラジオボタンで表示（視覚的フィードバック） ---
    
    # ラジオボタンの選択肢（Noneを許容しないため、indexで処理）
    point_options = list(range(len(labels)))
    st.session_state.active_point_index = st.radio(
        "💡 現在、設定したいポイントを選択してください:",
        point_options,
        index=st.session_state.active_point_index,
        format_func=lambda i: f"【{i+1}】 {labels[i]}",
        key=f'{coords_list_key}_active_point'
    )
    
    active_index = st.session_state.active_point_index
    active_label = labels[active_index]
    
    st.markdown("---")
    
    # --- 2. 選択中のポイントの入力欄だけをハイライト表示 ---
    
    st.subheader(f"✨ 設定中: {active_label} の座標")
    
    # 現在の値を取得 (設定済みの値、または初期値)
    initial_x, initial_y = current_coords[active_index]
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        x_val = st.number_input(
            f"X座標 (0-{image.shape[1]}): {active_label}",
            min_value=0,
            max_value=image.shape[1],
            value=initial_x,
            key=f'{coords_list_key}_x_{active_index}'
        )
    with col_y:
        y_val = st.number_input(
            f"Y座標 (0-{image.shape[0]}): {active_label}",
            min_value=0,
            max_value=image.shape[0],
            value=initial_y,
            key=f'{coords_list_key}_y_{active_index}'
        )

    st.markdown("---")

    # --- 3. 確定ボタン（クリックで次のポイントへ移動） ---
    
    if st.button(f"✅ {active_label} 座標を確定し、次のポイントへ", key=f'{coords_list_key}_confirm_btn'):
        handle_coordinate_input(
            coords_list_key, active_index, x_val, y_val, active_label
        )
    
    # 全てのポイントが設定済みか確認
    if all(coord is not None for coord in st.session_state[coords_list_key]):
        # 次のステップへ進むためのボタン
        st.success("全てのポイントが設定されました！")
        if st.button("次のステップへ進む", key=f'{coords_list_key}_next_step_btn'):
            if is_original_image:
                # ステップ1から2への移行（補正処理が必要）
                st.session_state.step = 2 
            else:
                # ステップ2から3への移行
                st.session_state.step = 3
            st.experimental_rerun()
            
    # 設定済みのポイントをマークしたデバッグ画像を準備
    display_debug_image(image, st.session_state[coords_list_key], labels)

def display_debug_image(image, coords_list, labels):
    if all(coord is not None for coord in coords_list):
        debug_image = image.copy()
        for i, (x, y) in enumerate(coords_list):
            x_int, y_int = int(x), int(y)
            # 現在アクティブなポイントを黄色で強調
            color = (0, 255, 255) if i == st.session_state.active_point_index else (0, 0, 255) 
            cv2.circle(debug_image, (x_int, y_int), 15 if i == st.session_state.active_point_index else 5, color, -1)
            cv2.putText(debug_image, f"{i+1}:{labels[i]}", (x_int + 15, y_int), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        st.subheader("現在の座標マーク（入力した座標を確認）")
        st.image(debug_image, channels="BGR", caption="設定したポイントを赤い丸で表示", use_column_width=True)


def main():
    init_session_state()

    st.title("📐 服のカスタム採寸アプリ (ガイド付き手動入力)")
    st.markdown("---")
    
    # ファイルアップローダー
    uploaded_file = st.file_uploader("ステップ 0: 画像をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if st.session_state.original_image is None or st.session_state.uploaded_file_name != uploaded_file.name:
            # 新しいファイルがアップロードされた場合、状態をリセット
            st.session_state.step = 1
            st.session_state.active_point_index = 0
            st.session_state.uploaded_file_name = uploaded_file.name
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.original_image = resize_image(original_image_bgr) # 処理用にリサイズ

    if st.session_state.original_image is None:
        st.info("縦51cm、横38cmの紙の上に服を置いて撮影した画像をアップロードしてください。")
        return

    # --- ステップ 1: 紙の角の指定とパースペクティブ補正 ---
    if st.session_state.step == 1:
        st.header("ステップ 1/3: 基準となる紙の4つの角を入力")
        st.warning("紙の角の正確な**X, Y座標**を入力してください。座標はPhotoshopなどのツールで確認できます。")
        st.info(f"紙の寸法: 縦{PAPER_HEIGHT_CM}cm, 横{PAPER_WIDTH_CM}cm")
        
        # 座標入力UIの呼び出し
        paper_labels = ["左上", "右上", "右下", "左下"]
        coordinate_input_ui(st.session_state.original_image, 'paper_coords', paper_labels, is_original_image=True)

    # --- ステップ 2: パースペクティブ補正と採寸点の指定 ---
    elif st.session_state.step == 2:
        st.header("ステップ 2/3: 採寸点の入力 (補正後画像)")

        # パースペクティブ補正を実行
        try:
            paper_points = np.array(st.session_state.paper_coords, dtype="float32")
            warped_image_bgr, _ = four_point_transform(
                st.session_state.original_image, paper_points, TARGET_WIDTH, TARGET_HEIGHT
            )
            st.session_state.processed_image = warped_image_bgr
            st.success("パースペクティブ補正が完了しました。以下の画像が補正後の画像です。")
        except Exception as e:
            st.error(f"パースペクティブ補正に失敗しました。ステップ1の座標を再確認してください: {e}")
            if st.button("← ステップ1に戻る", key="back_to_step1_from_2"):
                st.session_state.step = 1
                st.experimental_rerun()
            return

        # 座標入力UIの呼び出し
        measure_labels = ["着丈 (上端)", "着丈 (下端)", "身幅 (左端)", "身幅 (右端)"]
        coordinate_input_ui(st.session_state.processed_image, 'measure_coords', measure_labels, is_original_image=False)
        
        # 戻るボタン
        if st.button("← 紙の角の指定に戻る", key="back_to_step1_alt"):
            st.session_state.step = 1
            st.experimental_rerun()


    # --- ステップ 3: 結果の表示 ---
    elif st.session_state.step == 3:
        st.header("ステップ 3/3: 採寸結果")
        
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
            
            # --- 結果の視覚化 ---
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
            st.error(f"計測の計算中にエラーが発生しました。ステップ2の座標を再確認してください。: {e}")
            st.exception(e)
            
        # 戻るボタン
        if st.button("← ステップ2に戻って再調整する", key="back_to_step2_final"):
            st.session_state.step = 2
            st.experimental_rerun()

# アプリケーションの実行
if __name__ == '__main__':
    main()
