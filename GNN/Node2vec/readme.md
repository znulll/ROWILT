# Node2vec
### Paper
- Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.

### Concept
- Embed nodes with similar network neighborhoods close in the feature space
- Maximum likelihood optimization problem
- Independent to the downstream prediction task
- Biased walks
    - Trade off between local(BFS) and global(DFS)
    - Two parameters
        - Return parameter p
        - In-out parameter q
    - 2nd-order random walk : edge to edge