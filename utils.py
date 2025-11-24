# utils.py
import numpy as np
from scipy.spatial import distance as dist

def order_points(pts):
	# 4つの点（pts）を、[左上, 右上, 右下, 左下] の順に並び替える
	# 座標配列を初期化
	rect = np.zeros((4, 2), dtype="float32")

	# 1. 座標の合計に基づいて、左上 (最小) と右下 (最大) を特定
	# x + y が最小 -> 左上 (TL)
	# x + y が最大 -> 右下 (BR)
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)] # Top-Left (TL)
	rect[2] = pts[np.argmax(s)] # Bottom-Right (BR)

	# 2. 座標の差に基づいて、右上と左下を特定
	# y - x が最小 -> 右上 (TR)
	# y - x が最大 -> 左下 (BL)
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)] # Top-Right (TR)
	rect[3] = pts[np.argmax(diff)] # Bottom-Left (BL)

	# 並び替えられた座標を返す
	return rect
