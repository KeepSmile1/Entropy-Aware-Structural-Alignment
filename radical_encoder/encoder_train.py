import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import clip
# import cn_clip.clip as clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from clip_dataset import RadicalDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

# models = clip.available_models()
# print("支持的模型：", models)

# 定义训练函数
def train_clip(model, dataloader, epochs=15, lr=1e-6):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()

    # contrastive_loss = SupConLoss()
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    for epoch in range(epochs):
        total_loss = 0
        for original_images, augmented_images, texts in dataloader:
            original_images = original_images.to(device, dtype=torch.float32)
            augmented_images = augmented_images.to(device, dtype=torch.float32)
            texts = clip.tokenize(texts).to(device)

            # 前向传播（原图和增强图）
            # with torch.no_grad():
            original_features = model.encode_image(original_images)
            augmented_features = model.encode_image(augmented_images)
            text_features = model.encode_text(texts)

            original_features = original_features / (original_features.norm(dim=-1, keepdim=True) + 1e-8)
            augmented_features = augmented_features / (augmented_features.norm(dim=-1, keepdim=True) + 1e-8)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)

            # 原图和文本的余弦相似度
            logits_orig_text = original_features @ text_features.T / 0.07
            logits_aug_text = augmented_features @ text_features.T / 0.07

            # 合并损失
            ground_truth = torch.arange(len(original_images)).to(device)
            loss_orig = F.cross_entropy(logits_orig_text, ground_truth)
            loss_aug = F.cross_entropy(logits_aug_text, ground_truth)
            clip_loss = (loss_orig + loss_aug) / 2


            # con_loss = contrastive_loss(combined_features, labels=ground_truth)
            con_loss = contrastive_loss(original_features, augmented_features)

            triplet_loss_value = triplet_loss(
                anchor=original_features,
                positive=text_features,
                negative=text_features[torch.randperm(text_features.size(0))]
            )

            loss = clip_loss + 0.1 * con_loss + 1.0*triplet_loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
        print(f"clip_loss={clip_loss}, con_loss={con_loss}, triplet_loss={triplet_loss_value}")

    print("微调完成！")


def contrastive_loss(original_features, augmented_features, temperature=0.07):
    """
    参数：
        original_features: Tensor, [N, D]，原始图像的特征
        augmented_features: Tensor, [N, D]，增强图像的特征
        temperature: float, 温度系数 tau
    
    返回：
        loss: 对比损失标量
    """

    # original_features = F.normalize(original_features, dim=1)
    # augmented_features = F.normalize(augmented_features, dim=1)
    
    # logits[i, j] = dot(original_features[i], augmented_features[j]) / temperature
    logits = torch.matmul(original_features, augmented_features.t()) / temperature
    
    log_probs = F.log_softmax(logits, dim=1)  # 对每一行做 softmax
    
    loss_1 = -torch.mean(torch.diag(log_probs))
    
    # logits_t[i, j] = dot(augmented_features[i], original_features[j]) / temperature
    logits_t = torch.matmul(augmented_features, original_features.t()) / temperature
    log_probs_t = F.log_softmax(logits_t, dim=1)
    loss_2 = -torch.mean(torch.diag(log_probs_t))
    
    loss = (loss_1 + loss_2) / 2.0
    
    return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, features, labels=None):
            if len(features.shape) < 3:
                features = features.unsqueeze(1)
            batch_size = features.shape[0]  # Original batch size

            contrast_feature = torch.cat(features.unbind(dim=1), dim=0)
            anchor_dot_contrast = torch.matmul(
                contrast_feature, contrast_feature.T) / self.temperature

            logits_mask = torch.ones_like(anchor_dot_contrast)
            extended_batch_size = contrast_feature.shape[0]
            logits_mask = logits_mask - torch.eye(extended_batch_size).to(device)

            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().repeat(2, 2)

            exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
            log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

            assert mask.shape == log_prob.shape, f"Mask shape {mask.shape} != log_prob shape {log_prob.shape}"

            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)
            loss = -mean_log_prob_pos.mean()
            return loss

def compute_distribution_metrics(positive_scores, negative_scores, bins=50):
    """计算KL/JS散度及分布特征"""
    kde_pos = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(positive_scores.reshape(-1,1))
    kde_neg = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(negative_scores.reshape(-1,1))

    grid = np.linspace(-1, 1, bins).reshape(-1,1)
    p = np.exp(kde_pos.score_samples(grid))
    q = np.exp(kde_neg.score_samples(grid))
    p, q = p/p.sum()+1e-10, q/q.sum()+1e-10

    kl_pq = entropy(p, q)
    kl_qp = entropy(q, p)

    m = 0.5 * (p + q)
    js = 0.5 * (entropy(p, m) + entropy(q, m))
    
    return {"KL(P||Q)": kl_pq, "KL(Q||P)": kl_qp, "JS": js}

def evaluate_clip(model, dataloader):
    model.eval()
    total_similarities = []
    total_labels = []

    all_image_features = []  
    all_text_features = []   
    all_labels = []          

    all_texts = []
    for _, _, texts in dataloader:  
        all_texts.extend(texts)

    assert len(all_texts) == len(dataloader.dataset), "文本与图像数量不匹配"
    print(f"Extracted {len(all_texts)} unique texts for evaluation.") 

    text_tokens = clip.tokenize(all_texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化
        text_features = F.normalize(text_features, p=2, dim=-1)
        all_text_features = text_features 

    image_features_list = []
    with torch.no_grad():
        for images, _, _ in dataloader:
            images = images.to(device)
            image_features = model.encode_image(images)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = F.normalize(image_features, p=2, dim=-1)
            all_image_features.append(image_features.cpu().numpy())
            image_features_list.append(image_features)


    all_image_features_tensor = torch.cat(image_features_list, dim=0)
    with torch.no_grad():
        # image_features_all: [N, D], text_features: [N, D]
        similarities = (all_image_features_tensor @ all_text_features.T).cpu().numpy()
    print(f"sim is {similarities.shape}")
    print("Image features norm:", torch.norm(all_image_features_tensor, dim=1).mean().item())
    print("Text features norm:", torch.norm(all_text_features, dim=1).mean().item())
    mean_positive_similarity = np.mean(np.diag(similarities))
    visualize_similarity_distribution(similarities, save_path="similarity_distribution.png")
    print(f"Mean similarity for positives: {mean_positive_similarity:.4f}")

    # 计算 Top-k 准确率和 MRR
    top_k_results, mrr = evaluate_top_k_mrr(similarities)
    print("Top-k Accuracy:")
    for k, acc in top_k_results.items():
        print(f"  Top-{k}: {acc:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")


    # # 合并特征和标签
    # image_features = np.concatenate([feat.cpu().numpy() for feat in image_features_list], axis=0)
    # text_features = all_text_features.cpu().numpy()
    # labels = np.eye(len(dataloader.dataset), dtype=int)
    # # 计算马氏距离（图像侧
    # mahal_dist_image = evaluate_mahalanobis(image_features, labels.argmax(1))
    # print(f"Mahalanobis Distance (Image): {mahal_dist_image:.4f}")
    
    # # 计算KL/JS散度
    # positive_scores = similarities
    # negative_scores = similarities[~np.eye(similarities.shape[0], dtype=bool)]
    # dist_metrics = compute_distribution_metrics(positive_scores, negative_scores)
    # print(f"KL Divergence (P||Q): {dist_metrics['KL(P||Q)']:.4f}")
    # print(f"KL Divergence (Q||P): {dist_metrics['KL(Q||P)']:.4f}")
    # print(f"JS Divergence: {dist_metrics['JS']:.4f}")

def visualize_similarity_distribution(total_similarities, save_path="similarity_distribution.png"):
    """
    可视化正样本和负样本相似度的分布。
    """
    # 提取正样本相似度（对角线）
    positive_similarities = np.diag(total_similarities)

    # 提取负样本相似度（非对角线元素）
    rows, cols = total_similarities.shape
    if rows != cols:
        # 修正 total_similarities 为方阵
        min_dim = min(rows, cols)
        total_similarities = total_similarities[:min_dim, :min_dim]

    mask = ~np.eye(total_similarities.shape[0], dtype=bool)
    negative_similarities = total_similarities[mask]

    # 绘制正样本和负样本的直方图
    plt.figure(figsize=(12, 6))
    plt.hist(positive_similarities, bins=50, alpha=0.7, label='Positive Similarities', color='red', density=True)
    plt.hist(negative_similarities, bins=50, alpha=0.7, label='Negative Similarities', color='green', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Distribution of Cosine Similarities (Positive vs Negative)')
    plt.legend()
    plt.savefig(save_path.replace(".png", "_histogram.png"), bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.savefig(save_path.replace(".png", "_histogram.svg"), format='svg',bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

    # 绘制核密度估计图
    plt.figure(figsize=(12, 6))
    sns.kdeplot(positive_similarities, label='Positive Similarities', color='red', shade=True)
    sns.kdeplot(negative_similarities, label='Negative Similarities', color='green', shade=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('KDE of Cosine Similarities (Positive vs Negative)')
    plt.legend(loc='upper left')  # 图例放置在左上角
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.savefig(save_path.replace(".png", ".svg"), format='svg', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

def evaluate_top_k_mrr(similarity_matrix, top_k_values=[1, 5, 10]):
    """
    参数：
        similarity_matrix: numpy 数组，形状为 (N, N)，其中第 i 行对应图像 i 与所有文本的余弦相似度，
                           正确匹配假定在对角线位置（即第 i 个文本对应图像 i）。
        top_k_values: 列表，包含需要计算的 Top-k 值，例如 [1, 5, 10]

    返回：
        top_k_results: 字典，键为 k 值，值为 Top-k 准确率
        mrr: 平均倒数排名
    """
    # 检查是否为方阵
    rows, cols = similarity_matrix.shape
    if rows != cols:
        print("警告：相似度矩阵不是方阵，可能存在数据对齐问题！")
        min_dim = min(rows, cols)
        similarity_matrix = similarity_matrix[:min_dim, :min_dim]

    num_samples = similarity_matrix.shape[0]
    top_k_results = {k: 0 for k in top_k_values}
    reciprocal_ranks = []

    for i in range(num_samples):
        sorted_indices = np.argsort(-similarity_matrix[i])
        found_indices = np.where(sorted_indices == i)[0]
        if found_indices.size == 0:
            rank = num_samples
        else:
            rank = found_indices[0] + 1
        reciprocal_ranks.append(1.0 / rank)
        
        for k in top_k_values:
            if rank <= k:
                top_k_results[k] += 1

    top_k_results = {k: count / num_samples for k, count in top_k_results.items()}
    mrr = np.mean(reciprocal_ranks)
    
    return top_k_results, mrr

def evaluate_mahalanobis(features, labels):
    """计算马氏距离（需所有样本特征）"""
    cov = np.cov(features.T)
    inv_cov = np.linalg.pinv(cov)
    
    mean_pos = np.mean(features[labels==1], axis=0)
    mean_neg = np.mean(features[labels==0], axis=0)
    

    delta = mean_pos - mean_neg
    mahalanobis_dist = np.sqrt(delta.T @ inv_cov @ delta)
    return mahalanobis_dist


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = "cuda" if torch.cuda.is_available() else "cpu"
weight = None

# DataLoader
if __name__ == "__main__":
    root_dir = "train_radical_image_path" 
    dataset = RadicalDataset(root_dir=root_dir, csv_path='train_dataset_csv_path', transform=transform, augment=augment_transform,)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    print(f"Batch size: {dataloader.batch_size}")  
    print(f"Total batches: {len(dataloader)}")
    print(f"Total images: {len(dataloader.dataset)}")
    print("train数据集加载成功！")

    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.to(device).float()
    print("CLIP 模型加载成功！")
    train_clip(model, dataloader, epochs=20, lr=1e-5)

    # save model weight
    torch.save(model.state_dict(), weight)
    print("微调后的模型已保存！")


    weight = "save_path_weight"
    root_dir_source = "test_image_path"
    dataset_source = RadicalDataset(root_dir=root_dir_source, csv_path='test_dataset_csv_path', transform=transform, augment=augment_transform,)
    dataloader_source = DataLoader(dataset_source, batch_size=64, shuffle=False, drop_last=False)
    print(f"test_dataset load！{len(dataloader_source)}")

    print(f"Batch size: {dataloader_source.batch_size}")  
    print(f"Total batches: {len(dataloader_source)}")
    print(f"Total images: {len(dataloader_source.dataset)}")
    model_source, preprocess_source = clip.load("ViT-B/32", device=device)
    state_dict = torch.load(weight, map_location=device)
    model_source.load_state_dict(state_dict)

    evaluate_clip(model_source, dataloader_source)

