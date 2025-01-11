import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

def create_dataloaders(dataset, batch_size, train_ratio, bins, seed):
    labels = dataset.disease_tensor.cpu().numpy()
    indices = np.arange(len(labels))

    # bined
    bin_edges = np.histogram_bin_edges(labels, bins=bins)
    bin_indices = np.digitize(labels, bin_edges, right=True)

    train_indices, val_indices = [], []

    for bin_id in np.unique(bin_indices):
        bin_mask = bin_indices == bin_id  # 找到属于当前分箱的样本
        bin_data_indices = indices[bin_mask]  # 当前分箱样本的索引

        # 如果当前分箱不足2个样本，跳过该分箱
        if len(bin_data_indices) < 2:
            print(f"Bin {bin_id} has less than 2 samples ({len(bin_data_indices)} samples), adding to training set.")
            train_indices.extend(bin_data_indices)  # 将这些样本加入训练集
            continue

        # 按比例划分训练集和验证集
        bin_train_indices, bin_val_indices = train_test_split(
            bin_data_indices,
            train_size=train_ratio,
            random_state=seed
        )

        # 合并分箱内的索引到整体结果中
        train_indices.extend(bin_train_indices)
        val_indices.extend(bin_val_indices)

    # 转换为 numpy 数组
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    # 验证分布是否合理
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    print("---------------------------------------------")
    print(f"Train Mean: {np.mean(train_labels):.4f}, Std: {np.std(train_labels):.4f}")
    print(f"Val Mean: {np.mean(val_labels):.4f}, Std: {np.std(val_labels):.4f}")
    print(f"Number of Train Samples: {len(train_labels)}")
    print(f"Number of Validation Samples: {len(val_labels)}")
    print("---------------------------------------------")

    # 构建训练集和验证集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    return train_loader, val_loader