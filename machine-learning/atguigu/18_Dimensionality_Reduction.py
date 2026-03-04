import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

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
    生成两种演示数据集：
    1. 手写数字（64 维 → 降到 2 维，共 10 个类别）
    2. 瑞士卷（3 维流形嵌入，测试非线性降维效果）
    """
    # 1. 手写数字数据集（1797 张 8×8 图像，64 特征）
    digits = load_digits()
    X_digits = digits.data          # (1797, 64)
    y_digits = digits.target        # 0~9

    # 2. 瑞士卷数据集（3 维，测试流形学习）
    X_swiss, color_swiss = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)

    # 对数字数据集进行标准化（PCA 和 t-SNE 都推荐标准化）
    scaler = StandardScaler()
    X_digits_scaled = scaler.fit_transform(X_digits)

    return (X_digits_scaled, y_digits), (X_swiss, color_swiss)


# ============================================================
# 二、PCA 主成分分析
# ============================================================
def pca_demo(X, y):
    """
    PCA（主成分分析）演示：
    - 解释方差比（Explained Variance Ratio）—— 选择合适维度
    - 累积解释方差曲线
    - 2D / 3D 降维可视化（手写数字）
    - 重建误差分析（不同主成分数量）
    """
    print("\n" + "=" * 60)
    print("         PCA — 主成分分析（Principal Component Analysis）")
    print("=" * 60)

    # ---- 2.1 拟合全量 PCA，查看解释方差 ----
    pca_full = PCA(random_state=42)
    pca_full.fit(X)

    explained_var_ratio = pca_full.explained_variance_ratio_
    cumulative_var      = np.cumsum(explained_var_ratio)

    # 多少个主成分能解释 90% / 95% / 99% 的方差
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n_components = np.argmax(cumulative_var >= threshold) + 1
        print(f"  解释 {threshold*100:.0f}% 方差所需主成分数: {n_components}")

    # ---- 2.2 降维到 2D 和 3D ----
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d   = pca_2d.fit_transform(X)

    pca_3d = PCA(n_components=3, random_state=42)
    X_3d   = pca_3d.fit_transform(X)

    print(f"\n  原始维度      : {X.shape[1]}")
    print(f"  2D 降维后方差解释率: {pca_2d.explained_variance_ratio_.sum():.4f}")
    print(f"  3D 降维后方差解释率: {pca_3d.explained_variance_ratio_.sum():.4f}")

    # ---- 2.3 重建误差分析 ----
    print("\n  不同主成分数量的重建误差（MSE）：")
    n_components_list = [2, 5, 10, 20, 32, 64]
    mse_list = []
    for n in n_components_list:
        n = min(n, X.shape[1])
        pca_tmp = PCA(n_components=n, random_state=42)
        X_reduced    = pca_tmp.fit_transform(X)
        X_reconstructed = pca_tmp.inverse_transform(X_reduced)
        mse = np.mean((X - X_reconstructed) ** 2)
        mse_list.append(mse)
        print(f"    n_components={n:3d}  重建 MSE={mse:.6f}")

    # ---- 2.4 可视化 ----
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('PCA 主成分分析', fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 4, figure=fig)

    # 子图1：解释方差比（前 30 个主成分）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(1, 31), explained_var_ratio[:30], color='steelblue', alpha=0.8)
    ax1.set_xlabel('主成分编号')
    ax1.set_ylabel('单个方差解释率')
    ax1.set_title('各主成分解释方差比')
    ax1.grid(alpha=0.3)

    # 子图2：累积解释方差曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'b-', linewidth=2)
    for thr, color in zip([0.80, 0.90, 0.95, 0.99], ['orange', 'red', 'purple', 'brown']):
        n_comp = np.argmax(cumulative_var >= thr) + 1
        ax2.axhline(y=thr, color=color, linestyle='--', alpha=0.7,
                    label=f'{thr*100:.0f}%: {n_comp} 个')
        ax2.axvline(x=n_comp, color=color, linestyle=':', alpha=0.5)
    ax2.set_xlabel('主成分数量')
    ax2.set_ylabel('累积解释方差比')
    ax2.set_title('累积解释方差曲线')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # 子图3：重建误差
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot([min(n, X.shape[1]) for n in n_components_list], mse_list,
             'rs-', linewidth=2, markersize=6)
    ax3.set_xlabel('主成分数量')
    ax3.set_ylabel('重建误差（MSE）')
    ax3.set_title('主成分数量 vs 重建误差')
    ax3.grid(alpha=0.3)

    # 子图4：前两个主成分可视化（2D）
    ax4 = fig.add_subplot(gs[0, 3])
    scatter = ax4.scatter(X_2d[:, 0], X_2d[:, 1],
                          c=y, cmap='tab10', s=8, alpha=0.7)
    ax4.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    ax4.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    ax4.set_title('PCA 降维到 2D（手写数字）')
    plt.colorbar(scatter, ax=ax4, label='数字类别')

    # 子图5：前三个主成分可视化（3D）
    ax5 = fig.add_subplot(gs[1, :2], projection='3d')
    sc5 = ax5.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                      c=y, cmap='tab10', s=6, alpha=0.6)
    ax5.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
    ax5.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
    ax5.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
    ax5.set_title('PCA 降维到 3D（手写数字）')
    fig.colorbar(sc5, ax=ax5, label='数字类别', shrink=0.6)

    # 子图6：各类数字在 2D 空间的质心分布
    ax6 = fig.add_subplot(gs[1, 2:])
    for digit in range(10):
        mask = y == digit
        center = X_2d[mask].mean(axis=0)
        ax6.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, alpha=0.4, label=f'_{digit}')
        ax6.annotate(str(digit), center, fontsize=12, fontweight='bold',
                     color=plt.cm.tab10(digit / 10),
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    ax6.set_xlabel('PC1')
    ax6.set_ylabel('PC2')
    ax6.set_title('PCA 2D 空间中各数字类别分布')
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return X_2d, pca_2d


# ============================================================
# 三、t-SNE 降维
# ============================================================
def tsne_demo(X, y):
    """
    t-SNE（t-distributed Stochastic Neighbor Embedding）演示：
    - 不同 perplexity（困惑度）对结果的影响
    - 与 PCA 的直观对比
    - 局部结构保留 vs 全局结构
    """
    print("\n" + "=" * 60)
    print("         t-SNE（t-分布随机邻域嵌入）")
    print("=" * 60)
    print("  注意：t-SNE 计算较慢，请耐心等待...\n")

    # ---- 3.1 先用 PCA 预降维（加速 t-SNE，常见工程实践）----
    pca_pre = PCA(n_components=30, random_state=42)
    X_pca30 = pca_pre.fit_transform(X)
    print(f"  预降维：{X.shape[1]} 维 → PCA 30 维（保留方差: "
          f"{pca_pre.explained_variance_ratio_.sum():.4f}）")

    # ---- 3.2 对比不同 perplexity ----
    perplexity_list = [5, 30, 50]
    tsne_results = {}

    for perp in perplexity_list:
        print(f"  运行 t-SNE (perplexity={perp})...")
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=1000,
            learning_rate='auto',
            init='pca',           # 使用 PCA 初始化，收敛更稳定
            random_state=42
        )
        X_tsne = tsne.fit_transform(X_pca30)
        tsne_results[perp] = X_tsne
        print(f"    KL 散度: {tsne.kl_divergence_:.4f}")

    # ---- 3.3 可视化 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('t-SNE 不同 Perplexity 对比（手写数字）', fontsize=14, fontweight='bold')

    for ax, perp in zip(axes, perplexity_list):
        X_tsne = tsne_results[perp]
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                             c=y, cmap='tab10', s=8, alpha=0.8)
        ax.set_title(f'perplexity={perp}')
        ax.set_xlabel('t-SNE 维度 1')
        ax.set_ylabel('t-SNE 维度 2')
        plt.colorbar(scatter, ax=ax, label='数字类别')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

    return tsne_results[30]  # 返回 perplexity=30 的结果（最常用）


# ============================================================
# 四、PCA vs t-SNE 直观对比
# ============================================================
def comparison_demo(X, y, X_pca_2d, X_tsne_2d):
    """
    在手写数字数据集上直观对比 PCA 和 t-SNE：
    - PCA：线性降维，保留全局方差结构
    - t-SNE：非线性降维，保留局部邻域结构（类别分离更好）
    - 定量对比：类内距离 vs 类间距离
    """
    print("\n" + "=" * 60)
    print("         PCA vs t-SNE 对比分析")
    print("=" * 60)

    def compute_intra_inter_dist(X_2d, y):
        """计算平均类内距离和平均类间质心距离"""
        centers = np.array([X_2d[y == c].mean(axis=0) for c in range(10)])
        # 类内平均距离
        intra = np.mean([
            pairwise_distances(X_2d[y == c]).mean()
            for c in range(10)
        ])
        # 类间质心距离
        inter_dists = pairwise_distances(centers)
        np.fill_diagonal(inter_dists, np.nan)
        inter = np.nanmean(inter_dists)
        return intra, inter

    intra_pca,  inter_pca  = compute_intra_inter_dist(X_pca_2d,  y)
    intra_tsne, inter_tsne = compute_intra_inter_dist(X_tsne_2d, y)

    # 归一化，便于对比
    ratio_pca  = inter_pca  / intra_pca
    ratio_tsne = inter_tsne / intra_tsne

    print(f"\n{'指标':<20}{'PCA':>12}{'t-SNE':>12}")
    print("-" * 44)
    print(f"{'平均类内距离':<18}{intra_pca:>12.4f}{intra_tsne:>12.4f}")
    print(f"{'平均类间质心距离':<16}{inter_pca:>12.4f}{inter_tsne:>12.4f}")
    print(f"{'类间/类内 比值':<18}{ratio_pca:>12.4f}{ratio_tsne:>12.4f}  ← 越大越好")

    # ---- 可视化对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('PCA vs t-SNE 降维结果对比（手写数字，10 类）',
                 fontsize=14, fontweight='bold')

    titles = ['PCA（线性降维）', 't-SNE（非线性降维，perplexity=30）']
    results = [X_pca_2d, X_tsne_2d]
    x_labels = ['PC1', 't-SNE 维度 1']
    y_labels = ['PC2', 't-SNE 维度 2']

    for ax, title, X_2d, xl, yl in zip(axes, titles, results, x_labels, y_labels):
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                             c=y, cmap='tab10', s=10, alpha=0.7)
        # 标注每个类别的质心
        for digit in range(10):
            mask = y == digit
            center = X_2d[mask].mean(axis=0)
            ax.annotate(str(digit), center, fontsize=11, fontweight='bold',
                        color='white', ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.3',
                                  facecolor=plt.cm.tab10(digit / 10),
                                  alpha=0.85))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        plt.colorbar(scatter, ax=ax, label='数字类别')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# ============================================================
# 五、瑞士卷：非线性流形降维对比
# ============================================================
def swiss_roll_demo(X_swiss, color_swiss):
    """
    瑞士卷数据集演示：
    - 原始 3D 结构可视化
    - PCA（线性）：无法展开流形
    - t-SNE（非线性）：能较好地恢复流形结构
    """
    print("\n" + "=" * 60)
    print("         瑞士卷（Swiss Roll）流形降维对比")
    print("=" * 60)
    print("  运行 t-SNE（瑞士卷）...")

    # PCA 降维到 2D
    pca = PCA(n_components=2, random_state=42)
    X_swiss_pca = pca.fit_transform(X_swiss)
    print(f"  PCA 2D 累积方差解释率: {pca.explained_variance_ratio_.sum():.4f}")

    # t-SNE 降维到 2D
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                learning_rate='auto', init='pca', random_state=42)
    X_swiss_tsne = tsne.fit_transform(X_swiss)
    print(f"  t-SNE KL 散度: {tsne.kl_divergence_:.4f}")

    # ---- 可视化 ----
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('瑞士卷流形降维对比', fontsize=14, fontweight='bold')

    # 子图1：原始 3D 瑞士卷
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2],
                c=color_swiss, cmap='Spectral', s=8, alpha=0.8)
    ax1.set_title('原始瑞士卷（3D）')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 子图2：PCA 降维结果（线性方法，无法展开）
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_swiss_pca[:, 0], X_swiss_pca[:, 1],
                c=color_swiss, cmap='Spectral', s=8, alpha=0.8)
    ax2.set_title('PCA 降维（线性，无法展开流形）')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(alpha=0.3)

    # 子图3：t-SNE 降维结果（非线性，能学习流形结构）
    ax3 = fig.add_subplot(133)
    sc3 = ax3.scatter(X_swiss_tsne[:, 0], X_swiss_tsne[:, 1],
                      c=color_swiss, cmap='Spectral', s=8, alpha=0.8)
    ax3.set_title('t-SNE 降维（非线性，近似展开流形）')
    ax3.set_xlabel('t-SNE 维度 1')
    ax3.set_ylabel('t-SNE 维度 2')
    ax3.grid(alpha=0.3)
    plt.colorbar(sc3, ax=ax3, label='流形颜色编码')

    plt.tight_layout()
    plt.show()


# ============================================================
# 六、主函数
# ============================================================
def main():
    print("=" * 65)
    print("   无监督学习 —— 降维算法完整示例")
    print("   涵盖：PCA（主成分分析）/ t-SNE（非线性降维）")
    print("=" * 65)

    # 1. 生成数据
    (X_digits, y_digits), (X_swiss, color_swiss) = generate_datasets()
    print(f"\n数据集大小：")
    print(f"  手写数字（Digits）: {X_digits.shape}  类别数: {len(np.unique(y_digits))}")
    print(f"  瑞士卷（Swiss Roll）: {X_swiss.shape}")

    # 2. PCA 演示（手写数字）
    X_pca_2d, pca_model = pca_demo(X_digits, y_digits)

    # 3. t-SNE 演示（手写数字）
    X_tsne_2d = tsne_demo(X_digits, y_digits)

    # 4. PCA vs t-SNE 对比
    comparison_demo(X_digits, y_digits, X_pca_2d, X_tsne_2d)

    # 5. 瑞士卷流形降维对比
    swiss_roll_demo(X_swiss, color_swiss)

    print("\n" + "=" * 65)
    print("   示例运行完毕！")
    print("   核心要点总结：")
    print("   - PCA：线性，速度快，保留全局方差，适合预处理")
    print("   - t-SNE：非线性，保留局部结构，适合可视化探索")
    print("   - 实践中常先 PCA 降维，再用 t-SNE 精细可视化")
    print("=" * 65)


if __name__ == '__main__':
    main()
