"""对比实验结果可视化分析"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
csv_path = r'd:\学习\大创\Pedestrian_tracking\outputs\comparison\comparison_table.csv'
df = pd.read_csv(csv_path)

# 输出目录
output_dir = r'd:\学习\大创\Pedestrian_tracking\outputs\comparison\charts'
os.makedirs(output_dir, exist_ok=True)

# 定义方法和场景
methods = ['传统方法 (HOG+SVM)', '深度方法 (原始)', '深度方法 (微调)']
scenes = df['scene'].unique()

# 方法简称映射
method_short = {
    '传统方法 (HOG+SVM)': 'M1: 传统',
    '深度方法 (原始)': 'M2: 预训练',
    '深度方法 (微调)': 'M3: 微调'
}

# 颜色配置
colors = {
    '传统方法 (HOG+SVM)': '#FF6B6B',  # 红色
    '深度方法 (原始)': '#4ECDC4',     # 青色
    '深度方法 (微调)': '#45B7D1'      # 蓝色
}

# ========== 图1: 核心指标对比（MOTA, MOTP, IDF1）==========
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

metrics_core = ['MOTA', 'MOTP', 'IDF1']
for idx, metric in enumerate(metrics_core):
    ax = axes1[idx]
    x = np.arange(len(scenes))
    width = 0.25
    
    for i, method in enumerate(methods):
        data = df[df['method'] == method][metric].values
        bars = ax.bar(x + i * width, data, width, 
                     label=method_short[method], color=colors[method], alpha=0.8)
        # 添加数值标签
        for bar, val in zip(bars, data):
            ax.annotate(f'{val:.2f}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('场景', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'{metric} 对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.split('-')[1] for s in scenes], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'core_metrics_comparison.png'), dpi=150, bbox_inches='tight')
print(f"图1已保存: core_metrics_comparison.png")

# ========== 图2: ID切换与碎片化对比 ==========
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

metrics_track = ['IDSW', 'Frag']
for idx, metric in enumerate(metrics_track):
    ax = axes2[idx]
    x = np.arange(len(scenes))
    width = 0.25
    
    for i, method in enumerate(methods):
        data = df[df['method'] == method][metric].values
        bars = ax.bar(x + i * width, data, width, 
                     label=method_short[method], color=colors[method], alpha=0.8)
        for bar, val in zip(bars, data):
            ax.annotate(f'{int(val)}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('场景', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'{metric} 对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.split('-')[1] for s in scenes], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'tracking_stability.png'), dpi=150, bbox_inches='tight')
print(f"图2已保存: tracking_stability.png")

# ========== 图3: 检测性能对比（Precision, Recall）==========
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

metrics_det = ['Precision', 'Recall']
for idx, metric in enumerate(metrics_det):
    ax = axes3[idx]
    x = np.arange(len(scenes))
    width = 0.25
    
    for i, method in enumerate(methods):
        data = df[df['method'] == method][metric].values
        bars = ax.bar(x + i * width, data, width, 
                     label=method_short[method], color=colors[method], alpha=0.8)
        for bar, val in zip(bars, data):
            ax.annotate(f'{val:.2f}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('场景', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'{metric} 对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.split('-')[1] for s in scenes], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
fig3.savefig(os.path.join(output_dir, 'detection_performance.png'), dpi=150, bbox_inches='tight')
print(f"图3已保存: detection_performance.png")

# ========== 图4: 综合雷达图对比 ==========
fig4, ax4 = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

# 选择雷达图指标（归一化处理）
radar_metrics = ['MOTA', 'MOTP', 'IDF1', 'Precision', 'Recall']
# 对于IDSW和Frag，值越小越好，需要反转
# 计算各方法在所有场景的平均值
avg_data = df.groupby('method')[radar_metrics].mean()

# 归一化到0-1范围
normalized = avg_data.copy()
for col in radar_metrics:
    min_val = avg_data[col].min()
    max_val = avg_data[col].max()
    if max_val > min_val:
        normalized[col] = (avg_data[col] - min_val) / (max_val - min_val)
    else:
        normalized[col] = 1.0

# 绘制雷达图
angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合

for method in methods:
    values = normalized.loc[method].values.tolist()
    values += values[:1]
    ax4.plot(angles, values, 'o-', linewidth=2, label=method_short[method], color=colors[method])
    ax4.fill(angles, values, alpha=0.25, color=colors[method])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(radar_metrics, fontsize=11)
ax4.set_ylim(0, 1)
ax4.set_title('综合性能雷达图（归一化）', fontsize=14, fontweight='bold', y=1.08)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

fig4.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=150, bbox_inches='tight')
print(f"图4已保存: radar_comparison.png")

# ========== 图5: MOTA提升率对比 ==========
fig5, ax5 = plt.subplots(figsize=(10, 6))

# 计算相对于传统方法的MOTA提升率
baseline_mota = df[df['method'] == '传统方法 (HOG+SVM)']['MOTA'].values
mota_improvement = {}

for method in methods[1:]:  # 排除传统方法
    method_mota = df[df['method'] == method]['MOTA'].values
    # 提升率 = (新值 - 基准值) / |基准值| 或直接差值
    improvement = method_mota - baseline_mota
    mota_improvement[method] = improvement

x = np.arange(len(scenes))
width = 0.35

for i, method in enumerate(methods[1:]):
    bars = ax5.bar(x + i * width, mota_improvement[method], width,
                   label=method_short[method], color=colors[method], alpha=0.8)
    for bar, val in zip(bars, mota_improvement[method]):
        ax.annotate(f'+{val:.2f}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

ax5.set_xlabel('场景', fontsize=10)
ax5.set_ylabel('MOTA 提升值', fontsize=10)
ax5.set_title('MOTA 相对传统方法的提升', fontsize=12, fontweight='bold')
ax5.set_xticks(x + width/2)
ax5.set_xticklabels([s.split('-')[1] for s in scenes], fontsize=9)
ax5.legend(fontsize=10)
ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax5.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig5.savefig(os.path.join(output_dir, 'mota_improvement.png'), dpi=150, bbox_inches='tight')
print(f"图5已保存: mota_improvement.png")

# ========== 图6: FPS对比 ==========
fig6, ax6 = plt.subplots(figsize=(10, 6))

x = np.arange(len(scenes))
width = 0.25

for i, method in enumerate(methods):
    data = df[df['method'] == method]['FPS'].values
    bars = ax6.bar(x + i * width, data, width,
                   label=method_short[method], color=colors[method], alpha=0.8)
    for bar, val in zip(bars, data):
        ax.annotate(f'{val:.1f}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)

ax6.set_xlabel('场景', fontsize=10)
ax6.set_ylabel('FPS (帧/秒)', fontsize=10)
ax6.set_title('处理速度对比', fontsize=12, fontweight='bold')
ax6.set_xticks(x + width)
ax6.set_xticklabels([s.split('-')[1] for s in scenes], fontsize=9)
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig6.savefig(os.path.join(output_dir, 'fps_comparison.png'), dpi=150, bbox_inches='tight')
print(f"图6已保存: fps_comparison.png")

# ========== 生成分析报告 ==========
report_path = os.path.join(output_dir, 'analysis_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 对比实验结果分析报告\n\n")
    f.write("## 一、核心指标对比分析\n\n")
    
    # 计算平均值
    avg = df.groupby('method')[['MOTA', 'MOTP', 'IDF1', 'Precision', 'Recall', 'IDSW', 'Frag', 'FPS']].mean()
    
    f.write("### 各方法平均指标\n\n")
    f.write("| 方法 | MOTA | MOTP | IDF1 | Precision | Recall | IDSW | Frag | FPS |\n")
    f.write("|------|------|------|------|-----------|--------|------|------|-----|\n")
    for method in methods:
        row = avg.loc[method]
        f.write(f"| {method_short[method]} | {row['MOTA']:.3f} | {row['MOTP']:.3f} | {row['IDF1']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {int(row['IDSW'])} | {int(row['Frag'])} | {row['FPS']:.1f} |\n")
    
    f.write("\n### 关键发现\n\n")
    
    # MOTA分析
    f.write("#### 1. MOTA（多目标跟踪准确度）\n\n")
    f.write(f"- 传统方法平均MOTA: **{avg.loc['传统方法 (HOG+SVM)', 'MOTA']:.3f}**（负值，说明误检严重）\n")
    f.write(f"- 预训练模型平均MOTA: **{avg.loc['深度方法 (原始)', 'MOTA']:.3f}**\n")
    f.write(f"- 微调模型平均MOTA: **{avg.loc['深度方法 (微调)', 'MOTA']:.3f}**\n")
    f.write(f"- 微调相对传统方法提升: **+{avg.loc['深度方法 (微调)', 'MOTA'] - avg.loc['传统方法 (HOG+SVM)', 'MOTA']:.3f}**\n")
    f.write(f"- 微调相对预训练提升: **+{avg.loc['深度方法 (微调)', 'MOTA'] - avg.loc['深度方法 (原始)', 'MOTA']:.3f}**\n\n")
    
    # MOTP分析
    f.write("#### 2. MOTP（多目标跟踪精度）\n\n")
    f.write(f"- 传统方法平均MOTP: **{avg.loc['传统方法 (HOG+SVM)', 'MOTP']:.3f}**\n")
    f.write(f"- 深度方法平均MOTP: **{avg.loc['深度方法 (原始)', 'MOTP']:.3f}** ~ **{avg.loc['深度方法 (微调)', 'MOTP']:.3f}**\n")
    f.write("- 深度方法的定位精度显著优于传统方法（提升约0.2-0.3）\n\n")
    
    # IDF1分析
    f.write("#### 3. IDF1（身份识别F1分数）\n\n")
    f.write(f"- 传统方法平均IDF1: **{avg.loc['传统方法 (HOG+SVM)', 'IDF1']:.3f}**（几乎为0）\n")
    f.write(f"- 预训练模型平均IDF1: **{avg.loc['深度方法 (原始)', 'IDF1']:.3f}**\n")
    f.write(f"- 微调模型平均IDF1: **{avg.loc['深度方法 (微调)', 'IDF1']:.3f}**\n")
    f.write("- 深度方法在身份保持方面显著优于传统方法\n\n")
    
    # IDSW分析
    f.write("#### 4. IDSW（身份切换次数）\n\n")
    f.write(f"- 传统方法平均IDSW: **{int(avg.loc['传统方法 (HOG+SVM)', 'IDSW'])}**\n")
    f.write(f"- 预训练模型平均IDSW: **{int(avg.loc['深度方法 (原始)', 'IDSW'])}**\n")
    f.write(f"- 微调模型平均IDSW: **{int(avg.loc['深度方法 (微调)', 'IDSW'])}**\n")
    f.write("- 注意：微调模型的IDSW在MOT17-04场景显著增加（71次），可能因检测召回率提升导致更多关联挑战\n\n")
    
    # Precision/Recall分析
    f.write("#### 5. 检测性能（Precision vs Recall）\n\n")
    f.write(f"- 传统方法：Precision={avg.loc['传统方法 (HOG+SVM)', 'Precision']:.3f}, Recall={avg.loc['传统方法 (HOG+SVM)', 'Recall']:.3f}\n")
    f.write(f"- 预训练模型：Precision={avg.loc['深度方法 (原始)', 'Precision']:.3f}, Recall={avg.loc['深度方法 (原始)', 'Recall']:.3f}\n")
    f.write(f"- 微调模型：Precision={avg.loc['深度方法 (微调)', 'Precision']:.3f}, Recall={avg.loc['深度方法 (微调)', 'Recall']:.3f}\n")
    f.write("- **关键发现**：微调显著提升了Recall（从0.25提升到0.42），但Precision略有下降\n\n")
    
    # FPS分析
    f.write("#### 6. 处理速度（FPS）\n\n")
    f.write(f"- 传统方法平均FPS: **{avg.loc['传统方法 (HOG+SVM)', 'FPS']:.1f}**（CPU运行）\n")
    f.write(f"- 深度方法平均FPS: **{avg.loc['深度方法 (原始)', 'FPS']:.1f}** ~ **{avg.loc['深度方法 (微调)', 'FPS']:.1f}**（GPU运行）\n")
    f.write("- 深度方法速度提升约5倍\n\n")
    
    f.write("## 二、场景差异分析\n\n")
    
    for scene in scenes:
        scene_data = df[df['scene'] == scene]
        f.write(f"### {scene}\n\n")
        best_mota = scene_data.loc[scene_data['MOTA'].idxmax()]
        f.write(f"- 最佳MOTA: **{best_mota['method']}** ({best_mota['MOTA']:.3f})\n")
        best_idf1 = scene_data.loc[scene_data['IDF1'].idxmax()]
        f.write(f"- 最佳IDF1: **{best_idf1['method']}** ({best_idf1['IDF1']:.3f})\n\n")
    
    f.write("## 三、结论与建议\n\n")
    f.write("### 主要结论\n\n")
    f.write("1. **深度学习方法全面优于传统方法**：MOTA、MOTP、IDF1等核心指标均有显著提升\n")
    f.write("2. **LoRA微调有效**：微调模型在Recall和MOTA上优于预训练模型，证明领域适配有效\n")
    f.write("3. **检测召回率是关键瓶颈**：传统方法Recall仅0.02，深度方法提升到0.25-0.42\n")
    f.write("4. **ID切换问题仍需优化**：微调模型IDSW增加，说明高召回率带来更多关联挑战\n\n")
    
    f.write("### 改进建议\n\n")
    f.write("1. 进一步优化数据关联算法，降低ID切换次数\n")
    f.write("2. 增强ReID特征提取能力，提升遮挡后身份恢复\n")
    f.write("3. 平衡Precision与Recall，避免误检增加\n")

print(f"\n分析报告已保存: analysis_report.md")
print(f"\n所有图表已保存至: {output_dir}")