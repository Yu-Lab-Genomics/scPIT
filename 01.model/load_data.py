import torch

def load_data(data_path, device):
    try:
        # 加载张量
        loaded_tensors = torch.load(data_path)
        
        # 解包张量
        expr_tensor = loaded_tensors["expr_tensor"]
        expr_mask = loaded_tensors["expr_mask"]
        disease_tensor = loaded_tensors["fev1_tensor"]
        meta_tensor = loaded_tensors["meta_tensor"]
        celltype_tensor = loaded_tensors["celltype_tensor"]
        target_tensor = loaded_tensors["target"]

        print(f"expr_tensor{expr_tensor.shape}")
        print(f"expr_mask{expr_mask.shape}")
        print(f"disease_tensor{disease_tensor.shape}")
        print(f"meta_tensor{meta_tensor.shape}")
        print(f'celltype_tensor{celltype_tensor.shape}')
        print(f'target_tensor{target_tensor.shape}')
        print("---------------------------------------------")
        # 数据验证
        assert len(expr_tensor.shape) == 3, f"expr_tensor should be 3D, got shape {expr_tensor.shape}"
        assert len(expr_mask.shape) == 2, f"expr_mask should be 2D, got shape {expr_mask.shape}"
        assert len(disease_tensor.shape) == 1, f"disease_tensor should be 1D, got shape {disease_tensor.shape}"
        assert len(meta_tensor.shape) == 2, f"meta_tensor should be 2D, got shape {meta_tensor.shape}"
        
        # 确保维度匹配
        assert expr_tensor.shape[0] == expr_mask.shape[0] == disease_tensor.shape[0], \
            "Batch dimensions do not match"
        assert expr_tensor.shape[1] == expr_mask.shape[1], \
            "Cell dimensions do not match between expr_tensor and expr_mask"
        
        # 数据类型检查和转换
        expr_tensor = expr_tensor.float()  # 确保是float类型
        expr_mask = expr_mask.bool()      # mask转换为boolean类型
        disease_tensor = disease_tensor.float()  # 确保是float类型
        meta_tensor = meta_tensor.float() # 确保是float类型
        
        # 移到指定设备
        expr_tensor = expr_tensor.to(device)
        expr_mask = expr_mask.to(device)
        disease_tensor = disease_tensor.to(device)
        meta_tensor = meta_tensor.to(device)
        target_tensor = target_tensor.to(device)
        
        # 数据统计信息
        print("Data Statistics:")
        print(f"Number of samples: {expr_tensor.shape[0]}")
        print(f"Number of cells per sample: {expr_tensor.shape[1]}")
        print(f"Number of genes: {expr_tensor.shape[2]}")
        print(f"Valid cells ratio: {(~expr_mask).float().mean().item():.2%}")
        print(f"FEV1 range: [{disease_tensor.min().item():.2f}, {disease_tensor.max().item():.2f}]")
        print(f"meta_tensor: {meta_tensor.shape}")
        print(f"target_tensor: {target_tensor.shape}")

        return expr_tensor, expr_mask, disease_tensor, meta_tensor, celltype_tensor, target_tensor
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in loaded tensors: {e}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")