# Graph Neural Network

### Node Embedding
- Map nodes to d-dimensional embeddings such that similar nodes in the graph are embedded close together

### Downstream task
- Node classification
- Link prediction
- Community detection
- Network Similarity

### Setup
- V : Vertex set
- A : Adjacency matrix
- X : Node features matrix
- N(v) : The set of neighbors of v

### Difficulty of applying NN to graph directly
- Big O (Vertex)
- Not applicable to graphs of different sizes
- Sensitive to node ordering
- No fixed notion of locality or sliding window on the graph
- Graph is permutation invariant

### GNN
- 
#

### GCN
- Node's neighborhood defines a computation graph : Learn how to propagate information across the graph to compute node features
- Aggregate neighbors: 
    - Generate node embeddings based on local network neighborhoods using NN
    - Average information from neighbors and apply a neural network
- Train:
    - Supervised setting : Minimize the loss L
    - Unsupervised setting:
        - No node label available
        - Use the graph structure as the supervision
    - Node similarity
        - Random walk (node2vec, DeepWalk, struc2vec)
        - Matrix Factorization
        - Node proximity in the graph
 
#

### GraphSAGE
