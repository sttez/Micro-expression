#!/usr/bin/env python3
"""
完整的可视化和模型保存系统
训练完成后自动生成所有评估图表和报告
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ComprehensiveVisualizer:
    """
    综合可视化器

    这个类就像是一个专业的数据分析师，能够自动生成
    训练过程的各种分析图表和报告
    """

    def __init__(self, experiment_name: str = None):
        """
        初始化可视化器

        Args:
            experiment_name: 实验名称，如果为None则自动生成
        """

        # 如果没有提供实验名称，自动生成一个
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name

        # 创建实验目录结构，就像给每个实验建立独立的档案夹
        self.base_dir = Path("experiments") / experiment_name
        self.create_directory_structure()

        # 情感标签映射
        self.emotion_labels = {
            0: 'disgust',  # 厌恶
            1: 'happiness',  # 高兴
            2: 'others',  # 其他
            3: 'repression',  # 压抑
            4: 'surprise'  # 惊讶
        }

        self.emotion_chinese = {
            'disgust': '厌恶',
            'happiness': '高兴',
            'others': '其他',
            'repression': '压抑',
            'surprise': '惊讶'
        }

        # 颜色方案，让图表更美观
        self.colors = {
            'primary': '#2E86AB',  # 蓝色
            'secondary': '#A23B72',  # 紫色
            'success': '#F18F01',  # 橙色
            'danger': '#C73E1D',  # 红色
            'warning': '#F4D35E',  # 黄色
            'info': '#0EAD69'  # 绿色
        }

        # 初始化日志记录
        self.setup_logging()

    def create_directory_structure(self):
        """
        创建完整的目录结构

        这个方法就像是整理实验室的储物柜，为不同类型的
        结果文件创建专门的存放位置
        """

        directories = {
            'models': '保存训练好的模型文件',
            'plots': '保存所有图表和可视化结果',
            'logs': '保存训练日志和文本记录',
            'data': '保存处理后的数据和中间结果',
            'reports': '保存分析报告和总结文档',
            'config': '保存实验配置文件'
        }

        for dir_name, description in directories.items():
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

            # 在每个目录下创建说明文件
            readme_file = dir_path / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(f"# {dir_name.capitalize()}\n\n{description}\n")

        print(f"✅ 实验目录结构创建完成: {self.base_dir}")

    def setup_logging(self):
        """设置日志记录系统"""
        self.log_file = self.base_dir / "logs" / "experiment.log"

        # 写入实验开始时间和基本信息
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实验名称: {self.experiment_name}\n")
            f.write("=" * 60 + "\n\n")

    def log_message(self, message: str, level: str = "INFO"):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}\n"

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        print(f"📝 {message}")

    def save_model_with_metadata(self,
                                 model,
                                 results: Dict[str, Any],
                                 training_args: Dict[str, Any],
                                 optimizer_state: Optional[Dict] = None):
        """
        保存模型和所有相关元数据

        这个方法不仅保存模型权重，还保存训练过程中的
        所有重要信息，就像是制作一份完整的实验档案
        """

        self.log_message("开始保存模型和元数据...")

        # 保存完整的模型检查点
        model_path = self.base_dir / "models" / "final_model.pth"

        checkpoint = {
            'experiment_name': self.experiment_name,
            'save_time': datetime.now().isoformat(),
            'model_state_dict': model.state_dict(),
            'model_config': model.get_model_info() if hasattr(model, 'get_model_info') else {},
            'results': results,
            'training_args': training_args,
            'class_names': list(self.emotion_labels.values())
        }

        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, model_path)
        self.log_message(f"模型保存成功: {model_path}")

        # 保存模型配置的JSON文件（便于查看）
        config_path = self.base_dir / "config" / "model_config.json"
        model_config = {
            'experiment_name': self.experiment_name,
            'model_architecture': checkpoint.get('model_config', {}),
            'training_parameters': training_args,
            'final_results': {
                'accuracy': results.get('accuracy', 0),
                'total_epochs': len(results.get('history', {}).get('train_acc', [])),
                'best_val_acc': max(results.get('history', {}).get('val_acc', [0]))
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)

        return model_path

    def create_training_curves(self, history: Dict[str, List[float]]):
        """
        创建训练过程曲线图

        这些图表就像是训练过程的"心电图"，让我们能够
        直观地看到模型学习的整个过程
        """

        self.log_message("生成训练过程曲线图...")

        # 创建一个大的图表，包含多个子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'训练过程分析 - {self.experiment_name}',
                     fontsize=16, fontweight='bold')

        epochs = range(1, len(history['train_acc']) + 1)

        # 1. 准确率曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, history['train_acc'],
                 color=self.colors['primary'], linewidth=2,
                 label='训练准确率', marker='o', markersize=3)
        ax1.plot(epochs, history['val_acc'],
                 color=self.colors['danger'], linewidth=2,
                 label='验证准确率', marker='s', markersize=3)
        ax1.set_title('准确率变化曲线', fontweight='bold')
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('准确率 (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 标记最佳点
        best_val_epoch = np.argmax(history['val_acc']) + 1
        best_val_acc = max(history['val_acc'])
        ax1.axvline(x=best_val_epoch, color=self.colors['success'],
                    linestyle='--', alpha=0.7)
        ax1.annotate(f'最佳: {best_val_acc:.2f}%\n轮次: {best_val_epoch}',
                     xy=(best_val_epoch, best_val_acc),
                     xytext=(best_val_epoch + len(epochs) * 0.1, best_val_acc),
                     arrowprops=dict(arrowstyle='->', color=self.colors['success']),
                     fontsize=10, ha='left')

        # 2. 损失曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, history['train_loss'],
                 color=self.colors['primary'], linewidth=2,
                 label='训练损失', marker='o', markersize=3)
        ax2.plot(epochs, history['val_loss'],
                 color=self.colors['danger'], linewidth=2,
                 label='验证损失', marker='s', markersize=3)
        ax2.set_title('损失变化曲线', fontweight='bold')
        ax2.set_xlabel('训练轮数')
        ax2.set_ylabel('损失值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 学习率曲线（如果有的话）
        ax3 = axes[0, 2]
        if 'learning_rates' in history and history['learning_rates']:
            ax3.plot(epochs, history['learning_rates'],
                     color=self.colors['warning'], linewidth=2,
                     marker='d', markersize=3)
            ax3.set_title('学习率调度', fontweight='bold')
            ax3.set_xlabel('训练轮数')
            ax3.set_ylabel('学习率')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '无学习率数据', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=12)
            ax3.set_title('学习率调度', fontweight='bold')

        # 4. 过拟合分析
        ax4 = axes[1, 0]
        overfitting_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        ax4.plot(epochs, overfitting_gap,
                 color=self.colors['secondary'], linewidth=2,
                 marker='^', markersize=3)
        ax4.set_title('过拟合分析', fontweight='bold')
        ax4.set_xlabel('训练轮数')
        ax4.set_ylabel('训练-验证准确率差 (%)')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)

        # 添加过拟合警告线
        ax4.axhline(y=10, color='orange', linestyle=':', alpha=0.7,
                    label='轻度过拟合警戒线')
        ax4.axhline(y=20, color='red', linestyle=':', alpha=0.7,
                    label='严重过拟合警戒线')
        ax4.legend(fontsize=8)

        # 5. 训练稳定性分析
        ax5 = axes[1, 1]
        # 计算验证准确率的滑动标准差
        window_size = min(5, len(history['val_acc']) // 3)
        if window_size >= 2:
            val_acc_series = pd.Series(history['val_acc'])
            rolling_std = val_acc_series.rolling(window=window_size).std()
            ax5.plot(epochs, rolling_std,
                     color=self.colors['info'], linewidth=2,
                     marker='v', markersize=3)
            ax5.set_title(f'训练稳定性 ({window_size}轮滑动标准差)', fontweight='bold')
            ax5.set_xlabel('训练轮数')
            ax5.set_ylabel('验证准确率标准差')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, '数据不足\n无法分析稳定性', ha='center', va='center',
                     transform=ax5.transAxes, fontsize=12)
            ax5.set_title('训练稳定性分析', fontweight='bold')

        # 6. 性能总结
        ax6 = axes[1, 2]
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        best_val_acc = max(history['val_acc'])

        categories = ['最终\n训练准确率', '最终\n验证准确率', '最佳\n验证准确率']
        values = [final_train_acc, final_val_acc, best_val_acc]
        colors = [self.colors['primary'], self.colors['danger'], self.colors['success']]

        bars = ax6.bar(categories, values, color=colors, alpha=0.7)
        ax6.set_title('性能总结', fontweight='bold')
        ax6.set_ylabel('准确率 (%)')
        ax6.set_ylim(0, 100)

        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # 保存图表
        curves_path = self.base_dir / "plots" / "training_curves.png"
        plt.savefig(curves_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.log_message(f"训练曲线图保存成功: {curves_path}")
        return curves_path

    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        创建混淆矩阵可视化

        混淆矩阵就像是模型的"成绩单"，告诉我们模型在
        每个类别上的表现如何，哪些容易混淆
        """

        self.log_message("生成混淆矩阵图表...")

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'混淆矩阵分析 - {self.experiment_name}',
                     fontsize=16, fontweight='bold')

        class_names = [self.emotion_chinese[self.emotion_labels[i]] for i in range(5)]

        # 1. 原始数量混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax1, cbar_kws={'label': '样本数量'})
        ax1.set_title('混淆矩阵 (样本数量)', fontweight='bold')
        ax1.set_xlabel('预测标签', fontweight='bold')
        ax1.set_ylabel('真实标签', fontweight='bold')

        # 2. 归一化百分比混淆矩阵
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax2, cbar_kws={'label': '百分比'})
        ax2.set_title('归一化混淆矩阵 (百分比)', fontweight='bold')
        ax2.set_xlabel('预测标签', fontweight='bold')
        ax2.set_ylabel('真实标签', fontweight='bold')

        plt.tight_layout()

        # 保存图表
        cm_path = self.base_dir / "plots" / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        # 创建详细的混淆矩阵分析
        self.create_detailed_confusion_analysis(cm, cm_normalized, class_names)

        self.log_message(f"混淆矩阵图表保存成功: {cm_path}")
        return cm_path

    def create_detailed_confusion_analysis(self, cm, cm_normalized, class_names):
        """创建详细的混淆矩阵分析图表"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'详细混淆矩阵分析 - {self.experiment_name}',
                     fontsize=16, fontweight='bold')

        # 1. 每个类别的精确率
        ax1 = axes[0, 0]
        precision_scores = []
        for i in range(len(class_names)):
            if cm[:, i].sum() > 0:  # 避免除零
                precision = cm[i, i] / cm[:, i].sum()
            else:
                precision = 0
            precision_scores.append(precision)

        bars1 = ax1.bar(class_names, precision_scores, color=self.colors['primary'], alpha=0.7)
        ax1.set_title('各类别精确率', fontweight='bold')
        ax1.set_ylabel('精确率')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, score in zip(bars1, precision_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom')

        # 2. 每个类别的召回率
        ax2 = axes[0, 1]
        recall_scores = []
        for i in range(len(class_names)):
            if cm[i, :].sum() > 0:  # 避免除零
                recall = cm[i, i] / cm[i, :].sum()
            else:
                recall = 0
            recall_scores.append(recall)

        bars2 = ax2.bar(class_names, recall_scores, color=self.colors['danger'], alpha=0.7)
        ax2.set_title('各类别召回率', fontweight='bold')
        ax2.set_ylabel('召回率')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, score in zip(bars2, recall_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom')

        # 3. F1分数
        ax3 = axes[1, 0]
        f1_scores = []
        for p, r in zip(precision_scores, recall_scores):
            if p + r > 0:
                f1 = 2 * (p * r) / (p + r)
            else:
                f1 = 0
            f1_scores.append(f1)

        bars3 = ax3.bar(class_names, f1_scores, color=self.colors['success'], alpha=0.7)
        ax3.set_title('各类别F1分数', fontweight='bold')
        ax3.set_ylabel('F1分数')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, score in zip(bars3, f1_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom')

        # 4. 样本数量分布
        ax4 = axes[1, 1]
        sample_counts = cm.sum(axis=1)  # 每个类别的真实样本数
        bars4 = ax4.bar(class_names, sample_counts, color=self.colors['warning'], alpha=0.7)
        ax4.set_title('各类别样本数量', fontweight='bold')
        ax4.set_ylabel('样本数量')
        ax4.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for bar, count in zip(bars4, sample_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + max(sample_counts) * 0.01,
                     f'{count}', ha='center', va='bottom')

        plt.tight_layout()

        # 保存详细分析图表
        detailed_path = self.base_dir / "plots" / "detailed_confusion_analysis.png"
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.log_message(f"详细混淆矩阵分析保存成功: {detailed_path}")

    def create_performance_dashboard(self, results: Dict[str, Any]):
        """
        创建性能仪表板

        这个仪表板就像是汽车的仪表盘，一眼就能看到
        模型的关键性能指标
        """

        self.log_message("生成性能仪表板...")

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'模型性能仪表板 - {self.experiment_name}',
                     fontsize=20, fontweight='bold')

        # 获取关键指标
        accuracy = results.get('accuracy', 0)
        history = results.get('history', {})
        classification_rep = results.get('classification_report', {})

        # 1. 大号准确率显示
        ax1 = plt.subplot(3, 4, (1, 2))
        ax1.text(0.5, 0.5, f'{accuracy:.2f}%',
                 ha='center', va='center', fontsize=48, fontweight='bold',
                 color=self.colors['primary'], transform=ax1.transAxes)
        ax1.text(0.5, 0.2, '最终测试准确率',
                 ha='center', va='center', fontsize=16,
                 transform=ax1.transAxes)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')

        # 添加准确率等级评估
        if accuracy >= 80:
            grade = "优秀"
            grade_color = self.colors['success']
        elif accuracy >= 70:
            grade = "良好"
            grade_color = self.colors['warning']
        elif accuracy >= 60:
            grade = "及格"
            grade_color = self.colors['info']
        else:
            grade = "需改进"
            grade_color = self.colors['danger']

        ax1.text(0.5, 0.05, f'评级: {grade}',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 color=grade_color, transform=ax1.transAxes)

        # 2. 训练进度环形图
        ax2 = plt.subplot(3, 4, 3)
        if history:
            total_epochs = len(history['train_acc'])
            best_epoch = np.argmax(history['val_acc']) + 1

            # 创建简单的进度显示
            progress = best_epoch / total_epochs
            ax2.pie([progress, 1 - progress],
                    colors=[self.colors['success'], '#f0f0f0'],
                    startangle=90, counterclock=False)
            ax2.text(0, 0, f'{best_epoch}/{total_epochs}\n最佳轮次',
                     ha='center', va='center', fontsize=12, fontweight='bold')
        ax2.set_title('训练进度', fontweight='bold')

        # 3. 各类别性能雷达图
        ax3 = plt.subplot(3, 4, 4, projection='polar')
        if classification_rep:
            emotions = ['disgust', 'happiness', 'others', 'repression', 'surprise']
            f1_scores = []

            for emotion in emotions:
                if emotion in classification_rep:
                    f1_scores.append(classification_rep[emotion].get('f1-score', 0))
                else:
                    f1_scores.append(0)

            # 添加第一个点以闭合图形
            f1_scores.append(f1_scores[0])

            angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
            angles += angles[:1]  # 闭合

            ax3.plot(angles, f1_scores, 'o-', linewidth=2, color=self.colors['primary'])
            ax3.fill(angles, f1_scores, alpha=0.25, color=self.colors['primary'])
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels([self.emotion_chinese[e] for e in emotions])
            ax3.set_ylim(0, 1)
        ax3.set_title('各类别F1分数', fontweight='bold', pad=20)

        # 4-6. 训练历史的小图表
        if history:
            # 4. 准确率趋势
            ax4 = plt.subplot(3, 4, 5)
            epochs = range(1, len(history['train_acc']) + 1)
            ax4.plot(epochs, history['val_acc'], color=self.colors['primary'], linewidth=2)
            ax4.set_title('验证准确率趋势', fontweight='bold')
            ax4.set_xlabel('轮次')
            ax4.set_ylabel('准确率 (%)')
            ax4.grid(True, alpha=0.3)

            # 5. 损失趋势
            ax5 = plt.subplot(3, 4, 6)
            ax5.plot(epochs, history['val_loss'], color=self.colors['danger'], linewidth=2)
            ax5.set_title('验证损失趋势', fontweight='bold')
            ax5.set_xlabel('轮次')
            ax5.set_ylabel('损失值')
            ax5.grid(True, alpha=0.3)

            # 6. 学习曲线
            ax6 = plt.subplot(3, 4, 7)
            ax6.plot(epochs, history['train_acc'], label='训练',
                     color=self.colors['primary'], alpha=0.7)
            ax6.plot(epochs, history['val_acc'], label='验证',
                     color=self.colors['danger'], alpha=0.7)
            ax6.set_title('学习曲线对比', fontweight='bold')
            ax6.set_xlabel('轮次')
            ax6.set_ylabel('准确率 (%)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. 关键统计信息表格
        ax7 = plt.subplot(3, 4, (8, 12))
        ax7.axis('off')

        # 创建统计信息
        stats_data = []
        if history:
            stats_data.extend([
                ['训练轮数', f"{len(history['train_acc'])}"],
                ['最佳验证准确率', f"{max(history['val_acc']):.2f}%"],
                ['最终训练准确率', f"{history['train_acc'][-1]:.2f}%"],
                ['最终验证准确率', f"{history['val_acc'][-1]:.2f}%"],
                ['过拟合程度', f"{history['train_acc'][-1] - history['val_acc'][-1]:.2f}%"]
            ])

        stats_data.extend([
            ['测试准确率', f"{accuracy:.2f}%"],
            ['模型等级', grade]
        ])

        # 创建表格
        table = ax7.table(cellText=stats_data,
                          colLabels=['指标', '数值'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        # 设置表格样式
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor(self.colors['primary'])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
                cell.set_edgecolor('white')

        plt.tight_layout()

        # 保存仪表板
        dashboard_path = self.base_dir / "plots" / "performance_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        self.log_message(f"性能仪表板保存成功: {dashboard_path}")
        return dashboard_path

    def create_experiment_report(self,
                                 model,
                                 results: Dict[str, Any],
                                 training_args: Dict[str, Any],
                                 y_true: Optional[np.ndarray] = None,
                                 y_pred: Optional[np.ndarray] = None):
        """
        生成完整的实验报告

        这个报告就像是一份详细的研究论文，包含了
        实验的所有重要信息和分析结果
        """

        self.log_message("生成完整实验报告...")

        report_content = f"""
# 微表情识别实验报告

## 实验概要
- **实验名称**: {self.experiment_name}
- **生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- **实验类型**: 基于CNN+LSTM的微表情识别

## 模型架构
"""

        # 添加模型信息
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            report_content += f"""
### 模型参数统计
- **总参数量**: {model_info.get('total_parameters', 'N/A'):,}
- **可训练参数**: {model_info.get('trainable_parameters', 'N/A'):,}
- **冻结参数**: {model_info.get('frozen_parameters', 'N/A'):,}
- **序列长度**: {model_info.get('sequence_length', 'N/A')}
- **隐藏层1大小**: {model_info.get('hidden_size1', 'N/A')}
- **隐藏层2大小**: {model_info.get('hidden_size2', 'N/A')}
"""

        # 添加训练配置
        report_content += f"""
## 训练配置
"""
        for key, value in training_args.items():
            report_content += f"- **{key}**: {value}\n"

        # 添加训练结果
        history = results.get('history', {})
        if history:
            report_content += f"""
## 训练结果

### 基本指标
- **最终测试准确率**: {results.get('accuracy', 0):.4f}%
- **训练轮数**: {len(history.get('train_acc', []))}
- **最佳验证准确率**: {max(history.get('val_acc', [0])):.4f}%
- **最终训练准确率**: {history.get('train_acc', [0])[-1]:.4f}%
- **最终验证准确率**: {history.get('val_acc', [0])[-1]:.4f}%

### 训练分析
- **过拟合程度**: {history.get('train_acc', [0])[-1] - history.get('val_acc', [0])[-1]:.4f}%
- **训练稳定性**: {'良好' if len(history.get('val_acc', [])) > 0 and np.std(history['val_acc'][-10:]) < 5 else '一般'}
"""

        # 添加分类报告
        classification_rep = results.get('classification_report', {})
        if classification_rep:
            report_content += f"""
## 分类性能分析

### 各类别详细指标
"""
            for emotion in ['disgust', 'happiness', 'others', 'repression', 'surprise']:
                if emotion in classification_rep:
                    metrics = classification_rep[emotion]
                    chinese_name = self.emotion_chinese[emotion]
                    report_content += f"""
#### {emotion} ({chinese_name})
- **精确率**: {metrics.get('precision', 0):.4f}
- **召回率**: {metrics.get('recall', 0):.4f}
- **F1分数**: {metrics.get('f1-score', 0):.4f}
- **支持样本数**: {metrics.get('support', 0)}
"""

        # 添加实验总结
        accuracy = results.get('accuracy', 0)
        report_content += f"""
## 实验总结

### 性能评估
"""
        if accuracy >= 80:
            report_content += "- **总体评价**: 优秀 - 模型性能达到了很高的水平\n"
        elif accuracy >= 70:
            report_content += "- **总体评价**: 良好 - 模型性能符合预期\n"
        elif accuracy >= 60:
            report_content += "- **总体评价**: 及格 - 模型基本可用，但有改进空间\n"
        else:
            report_content += "- **总体评价**: 需改进 - 模型性能不理想，需要调优\n"

        report_content += f"""
### 改进建议
1. **数据方面**: 考虑增加更多训练数据，特别是表现较差的类别
2. **模型方面**: 可以尝试调整网络架构或超参数
3. **训练方面**: 可以尝试不同的优化策略或正则化方法

### 文件清单
- **模型文件**: models/final_model.pth
- **训练曲线**: plots/training_curves.png
- **混淆矩阵**: plots/confusion_matrix.png
- **性能仪表板**: plots/performance_dashboard.png
- **实验日志**: logs/experiment.log

### 实验环境
- **Python版本**: {os.sys.version.split()[0]}
- **PyTorch版本**: {torch.__version__}
- **CUDA可用**: {'是' if torch.cuda.is_available() else '否'}

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # 保存报告
        report_path = self.base_dir / "reports" / "experiment_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 同时保存HTML版本（如果可能）
        try:
            import markdown
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>实验报告 - {self.experiment_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #2E86AB; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{markdown.markdown(report_content)}
</body>
</html>
"""
            html_path = self.base_dir / "reports" / "experiment_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except ImportError:
            # markdown库不可用，跳过HTML生成
            pass

        self.log_message(f"实验报告保存成功: {report_path}")
        return report_path

    def save_training_log(self, training_output: str):
        """保存训练日志"""
        log_path = self.base_dir / "logs" / "training_output.log"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(training_output)

        self.log_message(f"训练日志保存成功: {log_path}")

    def create_complete_visualization_suite(self,
                                            model,
                                            results: Dict[str, Any],
                                            training_args: Dict[str, Any],
                                            y_true: Optional[np.ndarray] = None,
                                            y_pred: Optional[np.ndarray] = None,
                                            optimizer_state: Optional[Dict] = None):
        """
        创建完整的可视化套件

        这是主入口函数，会生成所有需要的图表和报告
        """

        self.log_message("开始创建完整的可视化套件...")

        created_files = []

        try:
            # 1. 保存模型和元数据
            model_path = self.save_model_with_metadata(
                model, results, training_args, optimizer_state
            )
            created_files.append(model_path)

            # 2. 创建训练过程曲线
            history = results.get('history', {})
            if history:
                curves_path = self.create_training_curves(history)
                created_files.append(curves_path)

            # 3. 创建混淆矩阵（如果有预测结果）
            if y_true is not None and y_pred is not None:
                cm_path = self.create_confusion_matrix(y_true, y_pred)
                created_files.append(cm_path)

            # 4. 创建性能仪表板
            dashboard_path = self.create_performance_dashboard(results)
            created_files.append(dashboard_path)

            # 5. 生成实验报告
            report_path = self.create_experiment_report(
                model, results, training_args, y_true, y_pred
            )
            created_files.append(report_path)

            # 6. 创建文件索引
            self.create_file_index(created_files)

            self.log_message("✅ 完整可视化套件创建完成!")

            # 打印总结信息
            self.print_summary()

            return self.base_dir

        except Exception as e:
            self.log_message(f"创建可视化套件时出错: {e}", "ERROR")
            raise

    def create_file_index(self, created_files: List[Path]):
        """创建文件索引"""
        index_content = f"""# 实验文件索引 - {self.experiment_name}

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 文件结构
```
{self.experiment_name}/
├── models/           # 模型文件
│   ├── final_model.pth
│   └── README.md
├── plots/            # 图表文件
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── detailed_confusion_analysis.png
│   └── performance_dashboard.png
├── logs/             # 日志文件
│   ├── experiment.log
│   └── training_output.log
├── reports/          # 报告文件
│   ├── experiment_report.md
│   └── experiment_report.html
├── data/             # 数据文件
└── config/           # 配置文件
    └── model_config.json
```

## 快速访问
- [实验报告](reports/experiment_report.md)
- [性能仪表板](plots/performance_dashboard.png)
- [训练曲线](plots/training_curves.png)
- [混淆矩阵](plots/confusion_matrix.png)
- [实验日志](logs/experiment.log)

## 使用说明
1. 查看 `reports/experiment_report.md` 了解完整实验结果
2. 查看 `plots/` 目录下的图表进行可视化分析
3. 使用 `models/final_model.pth` 进行推理预测
4. 参考 `config/model_config.json` 了解模型配置
"""

        index_path = self.base_dir / "README.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)

        self.log_message(f"文件索引创建成功: {index_path}")

    def print_summary(self):
        """打印实验总结"""
        print(f"""
{'=' * 60}
🎉 实验完成总结
{'=' * 60}

📁 实验名称: {self.experiment_name}
📂 保存路径: {self.base_dir}

📊 生成的文件:
  ✅ 模型文件: models/final_model.pth
  ✅ 训练曲线: plots/training_curves.png  
  ✅ 混淆矩阵: plots/confusion_matrix.png
  ✅ 性能仪表板: plots/performance_dashboard.png
  ✅ 实验报告: reports/experiment_report.md
  ✅ 实验日志: logs/experiment.log

🔍 快速查看:
  1. 查看完整报告: {self.base_dir}/reports/experiment_report.md
  2. 查看性能总览: {self.base_dir}/plots/performance_dashboard.png
  3. 分析训练过程: {self.base_dir}/plots/training_curves.png

💡 下次使用建议:
  - 使用保存的模型进行推理预测
  - 对比不同实验的结果
  - 根据分析结果调整超参数

{'=' * 60}
        """)


# 集成到主训练流程的函数
def integrate_with_training(model, results, training_args, test_loader, device, experiment_name=None):
    """
    与主训练流程集成的便捷函数

    在训练完成后调用这个函数，自动生成所有可视化结果
    """

    print("\n🎨 开始生成可视化结果...")

    # 创建可视化器
    visualizer = ComprehensiveVisualizer(experiment_name)

    # 评估模型获取预测结果
    print("📊 评估模型性能...")
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 创建完整的可视化套件
    experiment_dir = visualizer.create_complete_visualization_suite(
        model=model,
        results=results,
        training_args=training_args,
        y_true=y_true,
        y_pred=y_pred
    )

    return experiment_dir


# 使用示例
if __name__ == "__main__":
    # 这个脚本通常不会直接运行，而是被主训练程序调用
    print("""
    🎭 微表情识别可视化系统

    这个脚本提供了完整的训练结果可视化功能：

    ✨ 主要功能:
    - 自动创建实验目录结构
    - 生成训练过程曲线图
    - 创建混淆矩阵分析  
    - 生成性能仪表板
    - 制作完整实验报告
    - 保存所有相关文件

    📋 使用方法:
    在主训练程序中调用 integrate_with_training() 函数

    📁 输出结构:
    experiments/
    └── experiment_YYYYMMDD_HHMMSS/
        ├── models/      # 模型文件
        ├── plots/       # 图表文件  
        ├── logs/        # 日志文件
        ├── reports/     # 报告文件
        ├── data/        # 数据文件
        └── config/      # 配置文件
    """)