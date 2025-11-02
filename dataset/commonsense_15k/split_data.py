import json
import random

# 设置随机种子以保证可复现（可选）
random.seed(42)

# 读取原始 JSON 文件
with open('commonsense_15k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 打乱数据顺序
random.shuffle(data)

# 按比例划分
split_ratio = 0.7
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
test_data = data[split_index:]

# 保存到新的 JSON 文件
with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open('validation.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)
