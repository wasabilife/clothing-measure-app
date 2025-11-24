# utils.py
import numpy as np
from scipy.spatial import distance as dist

def order_points(pts):
    """
    4点の座標 (NumPy配列) を受け取り、左上、右上、右下、左下の順に並べ替える
    """
    # 座標を初期化
    rect = np.zeros((4, 2), dtype="float32")

    # 1. 左上と右下を特定: 座標の和が最小と最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 左上 (和が最小)
    rect[2] = pts[np.argmax(s)] # 右下 (和が最大)

    # 2. 右上と左下を特定: 座標の差が最小と最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上 (差が最小)
    rect[3] = pts[np.argmax(diff)] # 左下 (差が最大)

    return rect

def midpoint(ptA, ptB):
    """2点間の中点を計算する"""
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# 備考: measure_clothing関数内でも同様のロジックを実装可能ですが、
# 関数に分けることでコードが整理されます。
