import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming features is your [N, C] array and class_names is your list of class names
features = self.fisher_box_layer.data.cpu()   # shape [N, C]
features = features / (features.norm(dim=-1, p=2) + 1e-30).unsqueeze(-1)
class_names = cls_names  # length [N]

# Using t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=0)
reduced_features_tsne = tsne.fit_transform(features)

fig, ax = plt.subplots(dpi=1500)
plt.scatter(reduced_features_tsne[:, 0], reduced_features_tsne[:, 1])
for i, txt in enumerate(class_names):
    plt.annotate(txt, (reduced_features_tsne[i, 0], reduced_features_tsne[i, 1]), fontsize=1)
plt.title('t-SNE')
plt.savefig('./tsne.png')