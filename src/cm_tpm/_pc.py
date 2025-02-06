import numpy as np
from abc import ABC, abstractmethod

class PCNode(ABC):
    """Base class for all nodes in the Probabilistic Circuit."""
    def __init__(self):
        self.log_prob = None

    @abstractmethod
    def forward(self, X):
        """Compute the log-probability for input X."""
        pass

class SumNode(PCNode):
    """Sum node represents a mixture over child nodes."""
    def __init__(self, children, weights=None):
        super().__init__()
        self.children = children
        self.weights = weights if weights is not None else np.ones(len(children)) / len(children)

    def forward(self, X):
        return np.sum([w * child.forward(X) for w, child in zip(self.weights, self.children)], axis=0)
    
class ProductNode(PCNode):
    """Product node represents independent factorization over child nodes."""
    def __init__(self, children):
        super().__init__()
        self.children = children

    def forward(self, X):
        return np.prod([child.forward(X) for child in self.children], axis=0)
    
class LeafNode(PCNode):
    """Leaf node represents a distribution over the input variables."""
    def __init__(self, mean, var):
        super().__init__()
        self.mean = mean
        self.var = var

    def forward(self, X):
        return (1 / np.sqrt(2 * np.pi * self.var)) * np.exp(-0.5 * ((X - self.mean) ** 2) / self.var)
    

def build_tpm(X, max_depth=5):
    """Recursively builds a TPM structure from data."""
    if max_depth == 0 or X.shape[1] == 1:
        return LeafNode(np.mean(X), np.var(X) + 1e-6)  # Add small variance for stability

    # Ensure we don't create a ProductNode with only one child
    split = np.random.randint(1, X.shape[1])
    left_features = X[:, :split]
    right_features = X[:, split:]

    left_child = build_tpm(left_features, max_depth - 1)
    right_child = build_tpm(right_features, max_depth - 1)

    # Ensure ProductNodes always have at least 2 children
    if isinstance(left_child, LeafNode) and isinstance(right_child, LeafNode):
        return SumNode([left_child, right_child], weights=np.random.dirichlet(np.ones(2)))

    return ProductNode([left_child, right_child])  # Ensures at least two children


leaf1 = LeafNode(mean=5.0, var=4.0)
leaf2 = LeafNode(mean=10.0, var=9.0)

X_test = np.array([[5], [10], [0], [15]])

print(leaf1.forward(X_test))
print(leaf2.forward(X_test))

prod_node = ProductNode(children=[leaf1, leaf2])
print(prod_node.forward(X_test))

sum_node = SumNode(children=[leaf1, leaf2], weights=[0.5, 0.5])
print(sum_node.forward(X_test))


# Generate a small dataset
np.random.seed(42)
X = np.random.randn(100, 5)  # 100 samples, 5 features

# Construct a TPM with max depth 2
tpm_root = build_tpm(X, max_depth=2)

# Inspect structure recursively
def print_tpm_structure(node, depth=0):
    prefix = "  " * depth
    if isinstance(node, LeafNode):
        print(f"{prefix}LeafNode(mean={node.mean:.2f}, var={node.var:.2f})")
    elif isinstance(node, ProductNode):
        print(f"{prefix}ProductNode with {len(node.children)} children")
        for child in node.children:
            print_tpm_structure(child, depth + 1)
    elif isinstance(node, SumNode):
        print(f"{prefix}SumNode with {len(node.children)} mixtures")
        for child in node.children:
            print_tpm_structure(child, depth + 1)

print_tpm_structure(tpm_root)

log_likelihood = tpm_root.forward(X[:10])  # Check first 10 samples
print("Log-likelihoods:", log_likelihood)

def sample_tpm(node, n_samples=5):
    """Recursively samples from the TPM."""
    if isinstance(node, LeafNode):
        return np.random.normal(node.mean, np.sqrt(node.var), size=(n_samples, 1))
    elif isinstance(node, ProductNode):
        return np.hstack([sample_tpm(child, n_samples) for child in node.children])
    elif isinstance(node, SumNode):
        chosen_child = np.random.choice(node.children, p=node.weights)
        return sample_tpm(chosen_child, n_samples)

samples = sample_tpm(tpm_root, 10)
print("Generated Samples:\n", samples)