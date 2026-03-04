import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------------------------------------
# 中文字体 & 负号显示
# -----------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 一、生成示例数据集
# ============================================================
def generate_datasets():
    """
    生成三种经典的聚类测试数据集：
    1. 球形高斯团（适合 K-Means）
    2. 月牙形（适合 DBSCAN）
    3. 同心圆（适合 DBSCAN）
    """
    np.random.seed(42)

    # 1. 球形高斯团 —— 3 个真实簇
    X_blobs, y_blobs = make_blobs(
        n_samples=300,
        centers=[[-3, -3], [0, 3], [3, -1]],
        cluster_std=0.8,
        random_state=42
    )

    # 2. 月牙形数据
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

    # 3. 同心圆数据
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

    # 统一标准化
    scaler = StandardScaler()
    X_blobs   = scaler.fit_transform(X_blobs)
    X_moons   = scaler.fit_transform(X_moons)
    X_circles = scaler.fit_transform(X_circles)

    return (X_blobs, y_blobs), (X_moons, y_moons), (X_circles, y_circles)


# ============================================================
# 二、K-Means 聚类
# ============================================================
def kmeans_demo(X, true_labels=None):
    """
    K-Means 算法演示：
    - 使用肘部法则（Elbow Method）确定最优 K
    - 使用轮廓系数（Silhouette Score）辅助验证
    """
    print("\n" + "=" * 50)
    print("         K-Means 聚类算法")
    print("=" * 50)

    # ---- 2.1 肘部法则 ----
    print("\n--- 肘部法则选择最优 K ---")
    inertias = []          # 簇内误差平方和（SSE）
    silhouettes = []       # 轮廓系数
    k_range = range(2, 9)

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))
        print(f"  K={k}  SSE={km.inertia_:.2f}  轮廓系数={silhouettes[-1]:.4f}")

    # ---- 2.2 使用最优 K=3 拟合 ----
    best_k = 3
    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels_km = km_best.fit_predict(X)
    centers   = km_best.cluster_centers_

    print(f"\n最终选定 K={best_k}")
    print(f"SSE（簇内惯量）        : {km_best.inertia_:.4f}")
    print(f"轮廓系数 Silhouette    : {silhouette_score(X, labels_km):.4f}")
    print(f"Davies-Bouldin 指数   : {davies_bouldin_score(X, labels_km):.4f}  （越小越好）")
    print(f"Calinski-Harabasz 指数: {calinski_harabasz_score(X, labels_km):.4f}  （越大越好）")

    # ---- 2.3 可视化 ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('K-Means 聚类分析', fontsize=14, fontweight='bold')

    # 子图1：肘部法则
    ax = axes[0]
    ax.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=6)
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'最优 K={best_k}')
    ax.set_xlabel('簇数 K')
    ax.set_ylabel('SSE（簇内惯量）')
    ax.set_title('肘部法则（Elbow Method）')
    ax.legend()
    ax.grid(alpha=0.3)

    # 子图2：轮廓系数
    ax = axes[1]
    ax.plot(list(k_range), silhouettes, 'rs-', linewidth=2, markersize=6)
    ax.axvline(x=best_k, color='blue', linestyle='--', alpha=0.7, label=f'最优 K={best_k}')
    ax.set_xlabel('簇数 K')
    ax.set_ylabel('轮廓系数 Silhouette Score')
    ax.set_title('轮廓系数曲线')
    ax.legend()
    ax.grid(alpha=0.3)

    # 子图3：聚类结果
    ax = axes[2]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels_km, cmap='viridis', s=30, alpha=0.7)
    ax.scatter(centers[:, 0], centers[:, 1],
               c='red', marker='*', s=200, zorder=5, label='质心 Centroid')
    ax.set_title(f'K-Means 聚类结果 (K={best_k})')
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='簇编号')

    plt.tight_layout()
    plt.show()

    return labels_km


# ============================================================
# 三、DBSCAN 聚类
# ============================================================
def dbscan_demo(X_blobs, X_moons, X_circles):
    """
    DBSCAN（基于密度的噪声应用空间聚类）演示：
    - 对于非凸数据（月牙形、同心圆）表现优于 K-Means
    - 自动识别噪声点（标签为 -1）
    - 关键超参数：eps（邻域半径）、min_samples（最小点数）
    """
    print("\n" + "=" * 50)
    print("         DBSCAN 聚类算法")
    print("=" * 50)

    datasets = [
        (X_blobs,   "球形高斯团", 0.4, 5),
        (X_moons,   "月牙形数据", 0.3, 5),
        (X_circles, "同心圆数据", 0.3, 5),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('DBSCAN 聚类结果对比', fontsize=14, fontweight='bold')

    for ax, (X, title, eps, min_samples) in zip(axes, datasets):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        # 统计
        n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise     = np.sum(labels == -1)
        noise_ratio = n_noise / len(labels) * 100

        print(f"\n数据集: {title}")
        print(f"  eps={eps}, min_samples={min_samples}")
        print(f"  发现簇数: {n_clusters}")
        print(f"  噪声点数: {n_noise} ({noise_ratio:.1f}%)")
        if n_clusters > 1:
            core_mask  = labels != -1
            sil = silhouette_score(X[core_mask], labels[core_mask]) if core_mask.sum() > 1 else 0
            print(f"  轮廓系数 (剔除噪声): {sil:.4f}")

        # 绘制
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = 'black'   # 噪声点用黑色显示
            mask = labels == k
            label_name = f'噪声' if k == -1 else f'簇 {k}'
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[col], s=20, alpha=0.7, label=label_name)

        ax.set_title(f'{title}\n簇数={n_clusters}, 噪声={n_noise}点')
        ax.set_xlabel('特征 1')
        ax.set_ylabel('特征 2')
        ax.legend(fontsize=7, markerscale=1.5)

    plt.tight_layout()
    plt.show()


# ============================================================
# 四、层次聚类（Hierarchical Clustering）
# ============================================================
def hierarchical_demo(X):
    """
    层次聚类（凝聚型 Agglomerative Clustering）演示：
    - 通过树状图（Dendrogram）直观展示聚类过程
    - 支持多种连接方式：ward、complete、average、single
    """
    print("\n" + "=" * 50)
    print("         层次聚类（Hierarchical Clustering）")
    print("=" * 50)

    linkage_methods = ['ward', 'complete', 'average', 'single']

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('层次聚类分析', fontsize=14, fontweight='bold')

    # 使用 GridSpec 布局：上方树状图、下方聚类结果
    gs = gridspec.GridSpec(2, 4, figure=fig)

    print("\n各连接方式的评估指标（n_clusters=3）：")
    print(f"{'连接方式':<12} {'轮廓系数':>12} {'DB指数':>12} {'CH指数':>14}")
    print("-" * 52)

    for i, method in enumerate(linkage_methods):
        # ---- 4.1 使用 scipy 绘制树状图 ----
        ax_dendro = fig.add_subplot(gs[0, i])
        Z = linkage(X, method=method)
        dendrogram(Z, ax=ax_dendro, truncate_mode='level', p=4,
                   no_labels=True, color_threshold=0)
        ax_dendro.set_title(f'树状图\n({method})', fontsize=9)
        ax_dendro.set_xlabel('样本索引')
        ax_dendro.set_ylabel('距离')

        # ---- 4.2 sklearn 层次聚类 ----
        hc = AgglomerativeClustering(n_clusters=3, linkage=method)
        labels = hc.fit_predict(X)

        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
        print(f"{method:<12} {sil:>12.4f} {db:>12.4f} {ch:>14.4f}")

        # ---- 4.3 聚类结果散点图 ----
        ax_scatter = fig.add_subplot(gs[1, i])
        scatter = ax_scatter.scatter(X[:, 0], X[:, 1], c=labels,
                                     cmap='Set1', s=20, alpha=0.8)
        ax_scatter.set_title(f'聚类结果 ({method})', fontsize=9)
        ax_scatter.set_xlabel('特征 1')
        ax_scatter.set_ylabel('特征 2')

    plt.tight_layout()
    plt.show()


# ============================================================
# 五、算法综合对比
# ============================================================
def comparison_demo(X_blobs, X_moons, X_circles):
    """
    在三种数据集上横向对比 K-Means、DBSCAN、层次聚类的表现
    """
    print("\n" + "=" * 50)
    print("         三种算法综合对比")
    print("=" * 50)

    datasets = [
        (X_blobs,   "球形高斯团"),
        (X_moons,   "月牙形"),
        (X_circles, "同心圆"),
    ]

    algorithms = [
        ("K-Means\n(k=2/3/2)",
         [KMeans(n_clusters=3, n_init=10, random_state=42),
          KMeans(n_clusters=2, n_init=10, random_state=42),
          KMeans(n_clusters=2, n_init=10, random_state=42)]),
        ("DBSCAN\n(eps=0.3~0.4)",
         [DBSCAN(eps=0.4, min_samples=5),
          DBSCAN(eps=0.3, min_samples=5),
          DBSCAN(eps=0.3, min_samples=5)]),
        ("层次聚类 Ward\n(k=2/3/2)",
         [AgglomerativeClustering(n_clusters=3, linkage='ward'),
          AgglomerativeClustering(n_clusters=2, linkage='ward'),
          AgglomerativeClustering(n_clusters=2, linkage='ward')]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle('三种聚类算法在不同数据集上的对比', fontsize=13, fontweight='bold')

    for col, (algo_name, algo_list) in enumerate(algorithms):
        for row, ((X, ds_name), algo) in enumerate(zip(datasets, algo_list)):
            ax = axes[row][col]
            labels = algo.fit_predict(X)

            # 过滤噪声点后计算指标
            mask = labels != -1
            n_clusters = len(set(labels[mask]))

            if n_clusters > 1 and mask.sum() > 1:
                sil = silhouette_score(X[mask], labels[mask])
                subtitle = f'轮廓系数={sil:.3f}'
            else:
                subtitle = '（无法计算轮廓系数）'

            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)
            if row == 0:
                ax.set_title(f'{algo_name}', fontsize=9)
            if col == 0:
                ax.set_ylabel(ds_name, fontsize=9)
            ax.set_xlabel(subtitle, fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.show()


# ============================================================
# 六、主函数
# ============================================================
def main():
    print("=" * 60)
    print("   无监督学习 —— 聚类算法完整示例")
    print("   涵盖：K-Means / DBSCAN / 层次聚类")
    print("=" * 60)

    # 生成数据
    (X_blobs, y_blobs), (X_moons, y_moons), (X_circles, y_circles) = generate_datasets()
    print(f"\n数据集大小：球形高斯团={X_blobs.shape}, 月牙形={X_moons.shape}, 同心圆={X_circles.shape}")

    # K-Means —— 使用球形高斯团（最适合 K-Means）
    kmeans_demo(X_blobs)

    # DBSCAN —— 展示对非凸形状数据的优势
    dbscan_demo(X_blobs, X_moons, X_circles)

    # 层次聚类 —— 展示树状图与不同连接方式
    hierarchical_demo(X_blobs)

    # 综合对比
    comparison_demo(X_blobs, X_moons, X_circles)

    print("\n" + "=" * 60)
    print("   示例运行完毕！")
    print("=" * 60)


if __name__ == '__main__':
    main()
