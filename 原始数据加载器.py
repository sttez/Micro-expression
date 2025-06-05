import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from typing import List, Tuple, Dict
from collections import Counter


class CASME2Dataset(Dataset):
    """CASME2微表情数据集类"""

    def __init__(self,
                 root_dir: str,
                 excel_path: str,
                 sequence_length: int = 32,
                 mode: str = 'train',
                 transform=None,
                 augment_repeats: Dict[str, int] = None,
                 verbose: bool = True):
        """
        Args:
            root_dir: CASME2-RAW-cropped目录路径
            excel_path: 标注文件路径
            sequence_length: 序列长度
            mode: 'train' 或 'test'
            transform: 图像变换
            augment_repeats: 每个类别的重复次数字典
            verbose: 是否打印详细信息
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.mode = mode
        self.transform = transform
        self.verbose = verbose

        # 默认的数据增强重复次数（基于类别不平衡）
        if augment_repeats is None:
            self.augment_repeats = {
                'disgust': 8,
                'happiness': 16,
                'others': 5,
                'repression': 19,
                'surprise': 20
            }
        else:
            self.augment_repeats = augment_repeats

        # 标签映射
        self.label_map = {
            'disgust': 0,
            'happiness': 1,
            'others': 2,
            'repression': 3,
            'surprise': 4
        }

        # 加载数据
        self.sequences, self.original_counts = self._load_sequences(excel_path)

        if self.verbose:
            self._print_dataset_info()

    def _load_sequences(self, excel_path: str) -> Tuple[List[Dict], Dict]:
        """从Excel文件加载序列信息"""
        df = pd.read_excel(excel_path)
        sequences = []
        original_counts = Counter()

        for _, row in df.iterrows():
            subject = f"sub{int(row['Subject']):02d}"
            filename = row['Filename']
            emotion = row['Emotion']

            # 跳过未标注的情绪
            if emotion not in self.label_map:
                continue

            # 构建序列路径
            seq_path = os.path.join(self.root_dir, subject, filename)

            if os.path.exists(seq_path):
                # 获取所有帧
                frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.jpg')])

                if len(frames) >= self.sequence_length:
                    original_counts[emotion] += 1

                    # 根据模式和类别决定重复次数
                    repeats = self.augment_repeats[emotion] if self.mode == 'train' else 1

                    for repeat_idx in range(repeats):
                        sequences.append({
                            'path': seq_path,
                            'frames': frames,
                            'emotion': emotion,
                            'label': self.label_map[emotion],
                            'repeat_idx': repeat_idx,
                            'original_idx': len(sequences) // repeats if repeats > 1 else len(sequences)
                        })

        return sequences, dict(original_counts)

    def _print_dataset_info(self):
        """打印数据集信息"""
        print(f"\n=== {self.mode.upper()}数据集信息 ===")

        # 统计原始数据
        print("原始序列数量:")
        total_original = 0
        for emotion, count in self.original_counts.items():
            print(f"  {emotion}: {count}")
            total_original += count
        print(f"  总计: {total_original}")

        # 统计增强后数据
        if self.mode == 'train':
            print("\n增强后序列数量:")
            augmented_counts = Counter()
            for seq in self.sequences:
                augmented_counts[seq['emotion']] += 1

            total_augmented = 0
            for emotion, count in augmented_counts.items():
                original_count = self.original_counts[emotion]
                multiplier = count // original_count
                print(f"  {emotion}: {count} (原始: {original_count} × {multiplier})")
                total_augmented += count
            print(f"  总计: {total_augmented}")

            print(f"\n数据增强倍数: {total_augmented / total_original:.1f}x")

        print(f"序列长度: {self.sequence_length}")
        print(f"图像变换: {'启用' if self.transform else '未启用'}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """获取一个序列样本"""
        seq_info = self.sequences[idx]
        frames = seq_info['frames']

        # 随机选择起始帧（用于时间增强）
        start_idx = random.randint(0, len(frames) - self.sequence_length)
        selected_frames = frames[start_idx:start_idx + self.sequence_length]

        # 加载图像序列
        images = []
        for i, frame in enumerate(selected_frames):
            img_path = os.path.join(seq_info['path'], frame)
            img = Image.open(img_path).convert('L')  # 转为灰度图

            if self.transform:
                img = self.transform(img)

            images.append(img)

        # 转换为tensor (sequence_length, 1, H, W)
        images_tensor = torch.stack(images)

        return images_tensor, seq_info['label']


def get_transforms(mode='train', verbose=True):
    """获取数据变换"""
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),  # 随机旋转±5度
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 随机平移±5%
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
            transforms.Resize((128, 128)),  # 调整尺寸
            transforms.ToTensor(),  # 转换为tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图归一化
        ])

        if verbose:
            print("训练模式图像变换:")
            print("  - 随机旋转: ±5度")
            print("  - 随机仿射变换: 平移±5%")
            print("  - 随机水平翻转: 50%概率")
            print("  - 尺寸调整: 128×128")
            print("  - 归一化: mean=0.5, std=0.5")
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        if verbose:
            print("测试模式图像变换:")
            print("  - 尺寸调整: 128×128")
            print("  - 归一化: mean=0.5, std=0.5")

    return transform


def create_data_loaders(root_dir: str,
                        excel_path: str,
                        batch_size: int = 32,
                        sequence_length: int = 32,
                        train_split: float = 0.8,
                        num_workers: int = 4,
                        verbose: bool = True):
    """创建数据加载器"""

    if verbose:
        print("\n=== 创建数据加载器 ===")

    # 创建完整数据集（包含数据增强）
    full_dataset = CASME2Dataset(
        root_dir=root_dir,
        excel_path=excel_path,
        sequence_length=sequence_length,
        mode='train',
        transform=get_transforms('train', verbose),
        verbose=verbose
    )

    # 划分训练集和测试集
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size

    if verbose:
        print(f"\n数据集划分:")
        print(f"  总样本数: {total_size}")
        print(f"  训练集: {train_size} ({train_split * 100:.0f}%)")
        print(f"  测试集: {test_size} ({(1 - train_split) * 100:.0f}%)")

    # 使用固定种子确保可重复的划分
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    # 为测试集更换变换方式（去除数据增强）
    # 创建一个包装类来动态更改transform
    class TransformWrapper:
        def __init__(self, dataset, new_transform):
            self.dataset = dataset
            self.new_transform = new_transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # 获取原始数据集的样本
            original_transform = self.dataset.dataset.transform
            # 临时更换transform
            self.dataset.dataset.transform = self.new_transform
            # 获取样本
            sample = self.dataset[idx]
            # 恢复原始transform
            self.dataset.dataset.transform = original_transform
            return sample

    # 为测试集应用测试模式的变换
    test_transform = get_transforms('test', verbose)
    test_dataset_wrapped = TransformWrapper(test_dataset, test_transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    if verbose:
        print(f"\n数据加载器配置:")
        print(f"  批次大小: {batch_size}")
        print(f"  工作进程数: {num_workers}")
        print(f"  训练集批次数: {len(train_loader)}")
        print(f"  测试集批次数: {len(test_loader)}")
        print(f"  Pin Memory: {torch.cuda.is_available()}")
        print(f"  训练集使用数据增强: 是")
        print(f"  测试集使用数据增强: 否")

    return train_loader, test_loader


def analyze_dataset(excel_path: str, verbose: bool = True):
    """分析数据集分布"""
    df = pd.read_excel(excel_path)
    emotion_counts = df['Emotion'].value_counts()

    if verbose:
        print("完整数据集统计:")
        print("-" * 30)
        total_sequences = 0
        for emotion, count in emotion_counts.items():
            print(f"{emotion}: {count} sequences")
            total_sequences += count
        print("-" * 30)
        print(f"Total: {total_sequences} sequences")

        # 计算类别不平衡比例
        print(f"\n类别不平衡分析:")
        max_count = emotion_counts.max()
        for emotion, count in emotion_counts.items():
            ratio = max_count / count
            print(f"  {emotion}: {ratio:.1f}x 不平衡")

    return emotion_counts


def test_data_loading(root_dir: str, excel_path: str):
    """测试数据加载功能"""
    print("=== 数据加载测试 ===")

    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        root_dir=root_dir,
        excel_path=excel_path,
        batch_size=4,
        sequence_length=16,
        num_workers=0,  # 测试时使用0避免多进程问题
        verbose=True
    )

    # 测试加载一个批次
    print("\n测试加载训练数据...")
    for i, (sequences, labels) in enumerate(train_loader):
        print(f"批次 {i + 1}:")
        print(f"  序列形状: {sequences.shape}")  # (batch_size, seq_len, C, H, W)
        print(f"  标签形状: {labels.shape}")  # (batch_size,)
        print(f"  标签内容: {labels.tolist()}")

        if i >= 2:  # 只测试前3个批次
            break

    print("数据加载测试完成!")


# 使用示例
if __name__ == "__main__":
    # 数据路径（请根据实际情况修改）
    root_dir = "data/CASME2-RAW-cropped"
    excel_path = "data/CASME2-coding-20140508.xlsx"
    # root_dir = "data/Cropped"
    # excel_path = "data/CASME-coded-20190721.xls"
    # 分析数据集
    analyze_dataset(excel_path)

    # 测试数据加载（如果数据存在）
    if os.path.exists(root_dir) and os.path.exists(excel_path):
        test_data_loading(root_dir, excel_path)
    else:
        print("数据文件不存在，跳过数据加载测试")