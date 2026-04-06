import csv
import os
import lmdb
from PIL import Image
import six

def check_image_is_valid(image_path):
    """检查图像文件是否有效"""
    try:
        img = Image.open(image_path)
        img.verify()  # 验证图像文件是否有效
        return True
    except:
        return False

def create_lmdb_from_txt(txt_path, lmdb_path):
    """
    从 TXT 文件创建 LMDB 数据集
    :param txt_path: TXT 文件路径
    :param lmdb_path: LMDB 数据库保存路径
    """
    data_list = []
    with open(txt_path, 'r', encoding='utf-8') as txtfile:
        for line in txtfile:
            image_path, label = line.strip().split(' ', 1)  # 分割图像路径和标签
            data_list.append((image_path.strip(), label.strip()))

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
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
            except Exception as e:
                print(f"Error reading image data for {image_path}: {e}")
                continue # Skip to the next image if reading fails

            # 创建 LMDB 的键
            idx_str = f'{idx + 1:09d}'  # 索引格式为 9 位数字
            image_key = f'image-{idx_str}'.encode()
            label_key = f'label-{idx_str}'.encode()

            # 写入图像数据和标签
            txn.put(image_key, image_data)
            txn.put(label_key, label.encode('utf-8'))

    print(f"LMDB 数据集已创建：{lmdb_path}")
    env.close()

def validate_lmdb_dataset(lmdb_path):
    """验证 LMDB 数据集"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        num_samples = int(txn.get('num-samples'.encode()))
        print(f"样本总数：{num_samples}")

        os.makedirs('./tmp', exist_ok=True) # Ensure ./tmp exists
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
                try:
                    image = Image.open(six.BytesIO(image_data))
                    image_save_path = os.path.join('./tmp', f'{idx_str}_{label}.png')
                    image.save(image_save_path)
                    print(f"保存图片: {image_save_path}")
                except Exception as e:
                    print(f"Error saving image {idx} to disk: {e}")

            break

# example
if name == "__main__":
    txt_file_path = 'hwdb_train_500.txt' 
    lmdb_output_path = './lmdb_dataset_hwdb_train_500'  
    create_lmdb_from_txt(txt_file_path, lmdb_output_path)

    validate_lmdb_dataset(lmdb_output_path)