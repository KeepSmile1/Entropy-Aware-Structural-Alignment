import csv
import os
import lmdb
from PIL import Image

def check_image_is_valid(image_path):
    """检查图像文件是否有效"""
    try:
        img = Image.open(image_path)
        img.verify()  # 验证图像文件是否有效
        return True
    except:
        return False

def create_lmdb_from_csv(csv_path, lmdb_path):
    """
    从 CSV 文件创建 LMDB 数据集
    :param csv_path: CSV 文件路径
    :param lmdb_path: LMDB 数据库保存路径
    """
    data_list = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头
        for row in reader:
            image_path = row[0].strip()  # 第一列是图像路径
            label = row[2].strip()      # 第三列是标签
            data_list.append((image_path, label))

    # with open(csv_path, 'r', encoding='utf-8') as txtfile:
    #     for line in txtfile:
    #         image_path, label = line.strip().split(' ', 1)  # 分割图像路径和标签
    #         data_list.append((image_path.strip(), label.strip()))

    # 创建 LMDB 数据库
    env = lmdb.open(lmdb_path, map_size=1e12)  # 设置最大映射大小 1TB
    with env.begin(write=True) as txn:
        # 写入样本总数
        txn.put('num-samples'.encode(), str(len(data_list)).encode())
        
        for idx, (image_path, label) in enumerate(data_list):
            if not os.path.exists(image_path):
                print(f"图像路径不存在: {image_path}")
                continue
            
            # 验证图像是否有效
            if not check_image_is_valid(image_path):
                print(f"无效图像文件: {image_path}")
                continue

            # 读取图像数据并转换为二进制
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # 创建 LMDB 的键
            idx_str = f'{idx + 1:09d}'  # 索引格式为 9 位数字
            image_key = f'image-{idx_str}'.encode()
            label_key = f'label-{idx_str}'.encode()

            # 写入图像数据和标签
            txn.put(image_key, image_data)
            txn.put(label_key, label.encode('utf-8'))

    print(f"LMDB 数据集已创建：{lmdb_path}")
    env.close()

# example
# for csv_file_path in ['hwdb1x_10.csv',
#                     'hwdb1x_20.csv',
#                     'hwdb1x_30.csv',
#                     'hwdb1x_40.csv',
#                     'hwdb1x_50.csv',
#                 ]

# for num in [10,20,30,40,50]:
#     csv_file_path = f"icdar2013-test-{num}.csv"  # 替换为你的 CSV 文件路径
#     lmdb_output_path = f"./icdar2013-test-{num}"  # 替换为 LMDB 数据集保存路径
#     create_lmdb_from_csv(csv_file_path, lmdb_output_path)

if __name__ == "__main__":
    csv_file_paths = [
        # 'hwdb1x_train_500.csv',
        # 'hwdb1x_train_1500.csv',
        # 'hwdb1x_train_1000.csv',
        # 'hwdb1x_train_2000.csv',
        # 'hwdb1x_train_2755.csv',
        # 'hwdb1x_train_3755.csv',
        # 'icdar2013_train_standard_random_last_1000.csv'
    ]
    
    for csv_file_path in csv_file_paths:
        file_name = os.path.basename(csv_file_path).replace('.csv', '')
        lmdb_output_path = os.path.join(f'{file_name}')
        
        create_lmdb_from_csv(csv_file_path, lmdb_output_path)
        print(f"Generated LMDB file: {lmdb_output_path}")


import lmdb
from PIL import Image
import six

def validate_lmdb_dataset(lmdb_path):
    """验证 LMDB 数据集"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        num_samples = int(txn.get('num-samples'.encode()))
        print(f"样本总数：{num_samples}")

        for idx in range(1, num_samples + 1):
            idx_str = f'{idx:09d}'
            image_key = f'image-{idx_str}'.encode()
            label_key = f'label-{idx_str}'.encode()

            # 读取图像和标签
            image_data = txn.get(image_key)
            label = txn.get(label_key).decode('utf-8')

            print(f"样本 {idx}: 标签: {label}")
            # 将图像保存到本地
            if image_data:
                image = Image.open(six.BytesIO(image_data))
                image_save_path = os.path.join('./tmp', f'{idx_str}_{label}.png')
                image.save(image_save_path)
                print(f"保存图片: {image_save_path}")

# validate_lmdb_dataset('./lmdb_dataset_hwdb1test')