import math

# 输入输出文件路径
input_file = "radical_frequencies.txt"  # 部首频率文件
output_file = "radical_entropy.txt"  # 计算出的信息熵文件
def cal_entropy(input_file, output_file):
    frequency = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 2:
                radical = parts[0].strip()
                freq = int(parts[1].strip())
                frequency[radical] = freq

    total = sum(frequency.values())

    info_content = {}
    for radical, freq in frequency.items():
        p = freq / total
        info = -math.log(p) 
        info_content[radical] = info

    with open(output_file, "w", encoding="utf-8") as f:
        for radical, info in info_content.items():
            f.write(f"{radical}: {info:.6f}\n")

    print(f"信息熵计算完成，已保存到 {output_file}")


def heatmap_visual(entropy_data_path="radical_entropy.txt"):
    file_path = entropy_data_path
    characters = []
    entropy_values = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            char, entropy = line.strip().split(":")
            characters.append(char)
            entropy_values.append(float(entropy))

    assert len(entropy_values) == 400, f"数据的长度应该是400个字符，但实际是 {len(entropy_values)}"

    random.shuffle(entropy_values)
    entropy_matrix = np.array(entropy_values).reshape(20, 20)

    plt.figure(figsize=(10, 8))  # 调整图像大小
    sns.heatmap(entropy_matrix, annot=False, cmap="coolwarm", cbar=True, xticklabels=[], yticklabels=[], square=True)

    plt.title("Shuffled Character Entropy Heatmap (20x20 Matrix)")
    plt.show()


if name == "__main__":
    print("counting radical entropy...")
    cal_entropy(input_file, output_file)
    heatmap_visual(output_file)
    print("done")