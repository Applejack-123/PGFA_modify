import torch

# 加载文件
checkpoint = torch.load('split_1.pt', map_location='cpu')

# 查看类型
print(f"类型: {type(checkpoint)}")

# 如果是字典，查看所有键
if isinstance(checkpoint, dict):
    print(f"键: {checkpoint.keys()}")
    
    # 查看每个键对应的形状
    for key, value in checkpoint.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")