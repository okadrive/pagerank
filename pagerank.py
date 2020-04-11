import numpy as np

d = 0.1
ε = 0.00001
n = 6

# n 個の乱数を生成し配列に詰める
rand_array = np.array(np.random.rand(n))
r0 = 1/np.sum(rand_array) * rand_array
print("r0:", r0)

# 隣接行列 N
N = np.array([\
    [0,0,1,0,0,0], \
    [1,0,0,0,0,0], \
    [1,1,0,0,0,0], \
    [0,0,1,0,0,0], \
    [0,0,1,0,0,1], \
    [0,0,0,0,1,0]])
# 遷移行列 A1: n×nの空行列を作る
A1 = np.empty((n, n), float)
for i in range(n):
    column_sum = np.sum(N, axis=0)[i]
    # 外へのエッジを持たないノードの場合はランダムに他のノードへ遷移
    if column_sum == 0:
        A1[:,i:i+1] = np.array([[[1/n]] * n])
    else:
        A1[:,i:i+1] = np.array([[1/column_sum * N[:,i:i+1]]])

A2 = (1-d)*A1 + d/n * np.array([[1] * n])
r_prev = r0
while True:
    r_next = np.dot(A2, r_prev)
    # ノルムが1になるように正規化
    r_next = (np.sum(np.absolute(r_next) / np.sum(r_next))) * r_next
    if np.sum(np.absolute(r_next - r_prev)) < ε:
        np.set_printoptions(suppress=True, threshold=np.inf)
        # 閾値よりも抑えられたら r を PageRank と判定
        print("pR:", r_next)
        break
    # r を更新
    r_prev = r_next