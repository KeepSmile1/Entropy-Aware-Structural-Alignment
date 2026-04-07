import torch
import torch.nn as nn
import argparse
import os
import re
from collections import deque
from tqdm import tqdm


# =======================
# IDS Parsing
# =======================

IDS_OPERATORS = {'⿰', '⿱', '⿲', '⿳', '⿴', '⿵', '⿶', '⿷', '⿸', '⿹', '⿺', '⿻'}


def parse_ids(ids_sequence):
    """Parse IDS string into tree."""
    pattern = re.compile(r"&[A-Za-z0-9-]+;")

    if isinstance(ids_sequence, list):
        ids_sequence = ''.join(ids_sequence)

    tokens = []
    i = 0
    while i < len(ids_sequence):
        match = pattern.match(ids_sequence, i)
        if match:
            tokens.append(match.group())
            i = match.end()
        else:
            c = ids_sequence[i]
            if not c.isspace():
                tokens.append(c)
            i += 1

    idx = 0

    def parse_node():
        nonlocal idx
        symbol = tokens[idx]
        idx += 1

        if symbol in IDS_OPERATORS:
            node = {'symbol': symbol, 'children': []}
            num_children = 3 if symbol in {'⿲', '⿳'} else 2

            for _ in range(num_children):
                node['children'].append(parse_node())

            return node
        else:
            return {'symbol': symbol, 'children': []}

    return parse_node()


def get_parents(tree):
    """Build parent index list."""
    queue = deque([(tree, -1, 1, 0)])

    parents, nodes, depths, positions = [], [], [], []

    while queue:
        node, p, d, pos = queue.popleft()

        idx = len(nodes)
        parents.append(p)
        nodes.append(node["symbol"])
        depths.append(d)
        positions.append(pos)

        for i, child in enumerate(node["children"]):
            queue.append((child, idx, d + 1, min(i + 1, 2)))

    return parents, nodes, depths, positions

class TreePreprocessing(nn.Module):
    def __init__(self, clip_embeddings, device):
        super().__init__()
        self.clip_embeddings = clip_embeddings
        self.device = device

    def process(self, tree):
        nodes, depths, positions = self.traverse_tree(tree)
        parents, _ ,_ ,_ = get_parents(tree)

        # one-hot encodings
        # one_hot = torch.zeros(len(nodes), self.symbol_size)
        # for i, node in enumerate(nodes):
        #     index = self.symbol_map.get(node, self.symbol_map.get('unk'))
        #     one_hot[i][index] = 1

        embedding_dim = 512
        clip_encoded = torch.zeros((len(nodes), embedding_dim))
        for i, node in enumerate(nodes):
            clip_encoded[i] = self.clip_embeddings.get(node, torch.zeros(embedding_dim))
        for i, node in enumerate(nodes):
            if torch.all(clip_encoded[i] == 0) and node != "$":  # 检查是否为零向量
                print(f"Zero vector detected for node: {node}")

        depths = torch.tensor(depths, dtype=torch.float32)

        positions = torch.tensor(positions, dtype=torch.float32)
        tpe_positions = self.compute_parent_tree(depths, positions)

        tpe_child_positions = self.compute_child_tree(depths, positions, parents)

        return clip_encoded, depths, tpe_positions, tpe_child_positions

    def compute_parent_tree(self, depths, positions):
        D = 512
        d = torch.arange(1, D + 1, device=self.device)
        pi = torch.pi

        out = []
        for depth, pos in zip(depths, positions):
            if pos == 0:
                tpe = torch.sin(2 * d * pi / D)
            elif pos == 1:
                tpe = torch.sin((4 * depth - 2) * d * pi / D)
            else:
                tpe = torch.sin(4 * depth * d * pi / D)

            out.append(tpe)

        return torch.stack(out)

    def compute_child_tree(self, depths, positions, parents):
        tpe = self.compute_parent_tree(depths, positions)

        children = {}
        for i, p in enumerate(parents):
            if p != -1:
                children.setdefault(p, []).append(i)

        for i in reversed(range(len(depths))):
            if i in children:
                tpe[i] += tpe[children[i]].mean(dim=0)

        return tpe

    def traverse_tree(self, node, depth=1, pos=0, nodes=None, depths=None, positions=None):
        if nodes is None:
            nodes, depths, positions = [], [], []

        nodes.append(node['symbol'])
        depths.append(depth)
        positions.append(pos)

        for i, child in enumerate(node['children']):
            self.traverse_tree(child, depth + 1, min(i + 1, 2), nodes, depths, positions)

        return nodes, depths, positions


class DimensionDecomposition(nn.Module):
    def __init__(self, symbol_size, embedding_size):
        super(DimensionDecomposition, self).__init__()
        self.symbol_size = symbol_size
        self.embedding_size = embedding_size
        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(0.5, device='cuda'))
        self.beta = nn.Parameter(torch.tensor(0.001, device='cuda'))
    def forward(self, radical, depths, positions, tpe_child_positions=None):
        # Ensure depths and positions are column vectors
        depths = depths.unsqueeze(1)

        positions = positions.unsqueeze(1)
        
        # Dimension decomposition
        f_code = torch.sum(torch.pow(self.alpha, depths) * (1 - self.beta * positions) * radical, dim=0)   #乘法 or 幂 ？
        f_depth = torch.sum(depths * radical, dim=0) / (depths.max() + 1e-6)
        # f_parent = torch.sum(positions * radical, dim=0) / (positions.max() + 1e-6)

        tpe_reduced = torch.mean(positions, dim=1, keepdim=True)  # (N, 1)
        f_parent = torch.sum(tpe_reduced * radical, dim=0) / (tpe_reduced.max() + 1e-6)

        tpe_child_reduced = torch.mean(tpe_child_positions, dim=1, keepdim=True)  # (N, 1)
        f_child = torch.sum(tpe_child_reduced * radical, dim=0) / (tpe_child_reduced.max() + 1e-6)

        # Concatenate and apply MLP
        # embedding = torch.cat([f_code, f_depth, f_parent, f_child], dim=0)   # (N,3dx)
        # embedding = torch.cat([f_code, f_depth, f_parent], dim=0)
        # return embedding
        return f_code, f_depth, f_parent, f_child


class GateFusion(nn.Module):
    def __init__(self, input_size, output_size, num_features):
        """
        Gate Fusion 模块初始化
        :param input_size: 每个特征的维度，例如 512
        :param output_size: 融合后输出的维度
        :param num_features: 特征数量，例如 4
        """
        super(GateFusion, self).__init__()
        self.gate_fc = nn.Linear(num_features * input_size, num_features)  # 门控网络
        self.proj_fc = nn.Linear(input_size, output_size)  # 输出映射到目标维度
        # self.scale = nn.Parameter(torch.ones(output_size))  # 可学习的缩放参数
        # self.bias = nn.Parameter(torch.zeros(output_size))   # 可学习的平移参数

    def forward(self, features):
        """
        前向传播
        :param features: 一个包含 num_features 个特征的列表，每个特征形状为 [batch_size, input_size]
        :return: 融合后的特征，形状为 [batch_size, output_size]
        """
        # 拼接所有特征为 [batch_size, num_features * input_size]
        concatenated = torch.cat(features, dim=-1)
        gates = torch.sigmoid(self.gate_fc(concatenated))
        fused = sum(gate.unsqueeze(-1) * feature for gate, feature in zip(gates.unbind(dim=-1), features))

        output = self.proj_fc(fused)
        # output = output * self.scale + self.bias
        return output


class DVSRModel(nn.Module):
    def __init__(self, clip_embeddings, device, out_dim=1024):
        super().__init__()

        self.processor = TreePreprocessing(clip_embeddings, device)
        self.ddcm = DimensionDecomposition()
        self.fusion = GateFusion(512, out_dim)

    def forward(self, ids_batch):
        outputs = []

        for ids in ids_batch:
            tree = parse_ids(ids)
            x, d, p, tpe = self.processor.process(tree)

            h = self.ddcm(x, d, p, tpe)
            out = self.fusion(h)

            outputs.append(out)

        return torch.stack(outputs)


# =======================
# Feature Extraction
# =======================

def extract_features(model, char_ids_dict):
    results = {}

    model.eval()
    with torch.no_grad():
        for char, ids in tqdm(char_ids_dict.items()):
            out = model([ids])[0].cpu()
            results[char] = out.numpy()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_path", required=True)
    parser.add_argument("--clip_emb", required=True)
    parser.add_argument("--output", default="features.pt")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    clip_embeddings = torch.load(args.clip_emb, map_location=device)

    model = DVSRModel(clip_embeddings, device).to(device)

    # load IDS
    char_ids = {}
    with open(args.ids_path, encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                c, ids = line.strip().split(":", 1)
                char_ids[c] = ids

    print(f"Loaded {len(char_ids)} characters")

    feats = extract_features(model, char_ids)

    torch.save(feats, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()