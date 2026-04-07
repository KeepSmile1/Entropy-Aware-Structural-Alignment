import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties

CONFIG = {
    "embedding_path": "ft_clip_path.pt",
    "font_path": "TH-Sung-TP0.ttf",
    "n_clusters": 40,
    "show_annotations": True,

    # TSNE 参数
    "tsne": {
        "n_components": 2,
        "perplexity": 10,
        "early_exaggeration": 50,
        "learning_rate": 200,
        "n_iter": 1000,
        "random_state": 42
    },
    "save_path": "tsne_kmeans.png"
}



# filter unseen char
def filter_chars(clip_embeddings):
    unsupported_chars = [
        '𠁣', '𠃛', '𩰊', '𩰋'
    ]

    clip_char = [
        key for key in clip_embeddings.keys()
        if "&CDP-" not in key and key not in unsupported_chars
    ]

    return clip_char


def prepare_features(clip_embeddings, clip_char):
    features = torch.stack([
        torch.tensor(clip_embeddings[char])
        for char in clip_char
    ])

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    return features_normalized


def run_tsne(features, tsne_config):
    tsne = TSNE(**tsne_config)
    X_tsne = tsne.fit_transform(features)
    return X_tsne


def run_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels



def generate_colors(labels, num_colors=40):
    import matplotlib.cm as cm

    cmap = cm.get_cmap('gist_ncar', num_colors)
    colors = [cmap(label % num_colors) for label in labels]

    return colors


def visualize(X, labels, chars, config):
    font_prop = FontProperties(fname=config["font_path"])

    colors = generate_colors(labels, config["n_clusters"])

    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)

    # 是否标注字符
    if config["show_annotations"]:
        for i, char in enumerate(chars):
            try:
                plt.annotate(
                    char,
                    (X[i, 0], X[i, 1]),
                    fontproperties=font_prop,
                    fontsize=10,
                    ha='center',
                    va='center'
                )
            except Exception as e:
                print(f"无法渲染字符: {char}")

    plt.title("TSNE + KMeans Visualization", fontproperties=font_prop)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.tight_layout()
    plt.savefig(config["save_path"], dpi=300)
    plt.show()


def main():
    print("Loading embeddings...")
    clip_embeddings = torch.load(CONFIG["embedding_path"], map_location='cpu')

    print("Filtering characters...")
    clip_char = filter_chars(clip_embeddings)

    print(f"Total valid chars: {len(clip_char)}")

    print("Preparing features...")
    features = prepare_features(clip_embeddings, clip_char)

    print("Running TSNE...")
    X_tsne = run_tsne(features, CONFIG["tsne"])

    print("Running KMeans...")
    labels = run_kmeans(X_tsne, CONFIG["n_clusters"])

    print("Visualizing...")
    visualize(X_tsne, labels, clip_char, CONFIG)

    print("Done!")


if __name__ == "__main__":
    main()