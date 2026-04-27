# Core Abstractions: LouvainNE-Attributed

The system is structured as a training-free pipeline that transforms a raw attributed graph into low-dimensional embeddings via structural enrichment and hierarchical clustering.

## The Hybrid Graph ($\mathcal{G}_H$)
The central entity of the system is the **Hybrid Graph**. Unlike a standard graph $(\mathcal{V}, \mathcal{E})$, the Hybrid Graph $\mathcal{G}_H = (\mathcal{V}, \mathcal{E}_H, \mathbf{w})$ is a weighted undirected graph where $\mathcal{E}_H$ is the union of:
- **Structural Edges**: The original topological connections.
- **Attribute Edges**: Predicted connections based on node feature similarity.

The edge weight $w_{uv}$ represents the confidence of the connection, allowing the Louvain algorithm to prioritize high-signal paths.

## The Embedding Pipeline
The transformation from attributes $\mathbf{X}$ and structure $\mathcal{E}$ to embeddings $\mathbf{Z}$ follows a four-stage linear flow:

### 1. Attributed Fusion (The "Early Fusion" Layer)
This stage creates the Hybrid Graph. It uses **Mutual Top-K filtering**: an edge is added if node $u$ is among the top-K most similar nodes to $v$ AND $v$ is among the top-K most similar to $u$. This ensures high-confidence, symmetric connections and suppresses noise.

### 2. Hierarchical Embedding (The LouvainNE Core)
The system leverages the Louvain algorithm to recursively partition the Hybrid Graph. 
- **Community Hierarchy**: Each node is assigned a community ID at multiple levels of granularity.
- **Weighted Aggregation**: Final embeddings are produced by aggregating random vectors assigned to these communities, weighted by an exponential decay $\lambda$ (favoring coarser or finer structures based on tuning).

### 3. Sparse Neighbourhood Attention (The Refinement Layer)
To resolve boundaries and smooth representations, a single round of **Sparse Attention** is applied. Unlike dense attention, this is restricted to the immediate neighbourhood in $\mathcal{G}_H$, keeping complexity at $O(|\mathcal{E}_H|)$.

### 4. SVD Attribute-Residual (The Feature Bridge)
Because the community-detection path is lossy, the system maintains a parallel **SVD Branch**. A low-rank projection of the original feature matrix is concatenated to the structural embedding, ensuring the final representation preserves linear feature signals.

## Evaluation Ontology
To prevent "hallucinated" performance, the system implements a **Leakage-Free Protocol**:
- **Edge Partitioning**: Test edges are strictly removed *before* any part of the pipeline (including attribute-edge generation) begins.
- **Linear Probing**: Use of a simple Logistic Regression classifier ensures that the performance metrics reflect the *quality of the embeddings*, not the capacity of the classifier.
