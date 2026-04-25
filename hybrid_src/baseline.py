import networkx as nx
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def get_louvain_communities(G):
    """
    Computes Louvain communities using networkx.
    Falls back to nx.community if community (python-louvain) is missing.
    """
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        return partition
    except (ImportError, ModuleNotFoundError):
        print("python-louvain not found, falling back to networkx.algorithms.community.louvain_communities")
        from networkx.algorithms import community as nx_comm
        communities_list = list(nx_comm.louvain_communities(G))
        partition = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                partition[node] = i
        return partition

def majority_vote_prediction(partition, labels, train_mask):
    """
    Predicts label for each node based on community majority in THE TRAINING SET ONLY.
    """
    num_nodes = len(labels)
    # Convert masks to numpy if they are tensors
    if hasattr(train_mask, 'cpu'):
        train_mask = train_mask.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
        
    communities_train_labels = {}
    for node, comm_id in partition.items():
        if train_mask[node]:
            if comm_id not in communities_train_labels:
                communities_train_labels[comm_id] = []
            communities_train_labels[comm_id].append(labels[node])
    
    # Global majority for communities with no training nodes
    global_majority = Counter(labels[train_mask]).most_common(1)[0][0]
    
    comm_to_label = {}
    for comm_id in set(partition.values()):
        if comm_id in communities_train_labels:
            comm_to_label[comm_id] = Counter(communities_train_labels[comm_id]).most_common(1)[0][0]
        else:
            comm_to_label[comm_id] = global_majority
    
    preds = np.zeros(num_nodes)
    for node, comm_id in partition.items():
        preds[node] = comm_to_label[comm_id]
        
    return preds

def community_feature_classifier(partition, features, labels, train_mask):
    """
    Better baseline (Option B): Train a logistic regression for each community.
    If a community has too few training samples, fall back to global classifier.
    """
    num_nodes = len(labels)
    
    if hasattr(train_mask, 'cpu'):
        train_mask = train_mask.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    if hasattr(features, 'cpu'):
        features = features.cpu().numpy()

    # Train a global classifier as fallback
    global_clf = LogisticRegression(max_iter=1000, C=1.0)
    global_clf.fit(features[train_mask], labels[train_mask])
    
    # Group nodes by community
    comm_nodes = {}
    for node, comm_id in partition.items():
        if comm_id not in comm_nodes:
            comm_nodes[comm_id] = []
        comm_nodes[comm_id].append(node)
        
    preds = np.zeros(num_nodes)
    
    for comm_id, nodes in comm_nodes.items():
        comm_train_mask = [node for node in nodes if train_mask[node]]
        
        # We need at least 2 classes and enough samples to train a local classifier
        # In Cora, classes are 7. 
        if len(comm_train_mask) > 10 and len(set(labels[comm_train_mask])) > 1:
            try:
                local_clf = LogisticRegression(max_iter=500, C=1.0)
                local_clf.fit(features[comm_train_mask], labels[comm_train_mask])
                preds[nodes] = local_clf.predict(features[nodes])
            except:
                preds[nodes] = global_clf.predict(features[nodes])
        else:
            # Fallback to global classifier for this community
            preds[nodes] = global_clf.predict(features[nodes])
            
    return preds

def get_community_label_distribution(partition, labels, train_mask, num_classes):
    """
    Returns soft probabilities for each node based on its community's training label distribution.
    Used for Soft Fusion.
    """
    num_nodes = len(labels)
    if hasattr(train_mask, 'cpu'): train_mask = train_mask.cpu().numpy()
    if hasattr(labels, 'cpu'): labels = labels.cpu().numpy()
        
    comm_train_counts = {}
    for node, comm_id in partition.items():
        if train_mask[node]:
            if comm_id not in comm_train_counts:
                comm_train_counts[comm_id] = Counter()
            comm_train_counts[comm_id][labels[node]] += 1
    
    # Global distribution for empty communities
    global_counts = Counter(labels[train_mask])
    global_total = sum(global_counts.values())
    global_dist = np.zeros(num_classes)
    for cls, count in global_counts.items():
        global_dist[cls] = count / global_total
        
    comm_to_dist = {}
    communities = set(partition.values())
    for comm_id in communities:
        if comm_id in comm_train_counts:
            c_counts = comm_train_counts[comm_id]
            total = sum(c_counts.values())
            d = np.zeros(num_classes)
            for cls, count in c_counts.items():
                d[cls] = count / total
            comm_to_dist[comm_id] = d
        else:
            comm_to_dist[comm_id] = global_dist
            
    probs = np.zeros((num_nodes, num_classes))
    for node, comm_id in partition.items():
        probs[node] = comm_to_dist[comm_id]
        
    return probs

def learned_confidence(G, partition, preds, labels, train_mask):
    """
    Learns 'where to learn' using high-fidelity signals:
    Input:
    - Structural: Degree, Consistency
    - Semantic: Community Label Entropy, Neighbor Label Diversity
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import entropy
    import numpy as np
    from collections import Counter
    
    num_nodes = len(preds)
    
    # 1. Community-level Semantic Signals
    comm_labels = {cid: [] for cid in set(partition.values())}
    for i in range(num_nodes):
        comm_labels[partition[i]].append(preds[i])
        
    comm_entropies = {}
    for cid, lbls in comm_labels.items():
        counts = list(Counter(lbls).values())
        comm_entropies[cid] = entropy(counts) if len(counts) > 1 else 0
    
    # 2. Node-level Feature Engineering
    deg = dict(G.degree())
    features = []
    
    for i in range(num_nodes):
        neighbors = list(G.neighbors(i))
        # Structural 1: Degree
        d = deg[i]
        # Semantic 1: Community Label Entropy
        c_ent = comm_entropies[partition[i]]
        
        if not neighbors:
            features.append([d, 1.0, 1.0, c_ent, 0.0])
            continue
            
        # Structural 2: Internal/External Ratio
        internal = sum(1 for n in neighbors if partition[n] == partition[i])
        consistency = internal / d
        
        # Semantic 2: Neighbor Agreement
        agree = sum(1 for n in neighbors if preds[n] == preds[i])
        agreement = agree / d
        
        # Semantic 3: Neighbor Label Diversity
        neigh_labels = [preds[n] for n in neighbors]
        neigh_counts = list(Counter(neigh_labels).values())
        neigh_ent = entropy(neigh_counts) if len(neigh_counts) > 1 else 0
        
        features.append([d, consistency, agreement, c_ent, neigh_ent])
        
    features = np.array(features)
    
    # 2. Training Labels (is Louvain correct?)
    # Only training on nodes where we have ground truth
    train_indices = np.where(train_mask)[0]
    y_train = (preds[train_indices] == labels[train_indices]).astype(int)
    X_train = features[train_indices]
    
    # 3. Fit Confidence Predictor
    if len(set(y_train)) > 1:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        # Probability of being correct
        confidences = clf.predict_proba(features)[:, 1]
    else:
        # Fallback to heuristic
        print("WARNING: Learned confidence failed (single class). Using heuristic.")
        confidences = compute_confidence(G, partition, preds)
        
    # ADVANCED FIX: Uncertainty Propagation
    # conf_final = alpha * self_conf + (1-alpha) * mean(neighbor_conf)
    alpha = 0.7
    final_confs = np.zeros(num_nodes)
    for i in range(num_nodes):
        neighbors = list(G.neighbors(i))
        if not neighbors:
            final_confs[i] = confidences[i]
            continue
        neigh_conf = np.mean([confidences[n] for n in neighbors])
        final_confs[i] = alpha * confidences[i] + (1 - alpha) * neigh_conf
        
    return final_confs

def compute_confidence(G, partition, preds):
    """
    Improved confidence metric:
    confidence(node) = 0.5 * neighbor_agreement + 0.5 * community_purity
    
    neighbor_agreement: % of neighbors with same predicted label.
    community_purity: % of nodes in the same community with same predicted label.
    """
    confidences = np.zeros(len(preds))
    num_nodes = len(preds)
    
    # Precompute community purities
    comm_nodes = {}
    for node, comm_id in partition.items():
        if comm_id not in comm_nodes:
            comm_nodes[comm_id] = []
        comm_nodes[comm_id].append(node)
        
    comm_purity = {}
    for comm_id, nodes in comm_nodes.items():
        labels_in_comm = [preds[n] for n in nodes]
        label_counts = Counter(labels_in_comm)
        # Fraction of nodes in community that have the majority label (which is preds[node])
        # Note: Since preds[node] is the same for all nodes in the community for majority baseline,
        # purity is constant for the whole community.
        majority_label = label_counts.most_common(1)[0][0]
        comm_purity[comm_id] = label_counts[majority_label] / len(nodes)

    for node in G.nodes():
        # 1. Neighbor Agreement
        neighbors = list(G.neighbors(node))
        if not neighbors:
            agreement = 1.0
        else:
            agree_count = sum(1 for n in neighbors if preds[n] == preds[node])
            agreement = agree_count / len(neighbors)
            
        # 2. Community Purity
        purity = comm_purity[partition[node]]
        
        # Combined Score
        confidences[node] = 0.5 * agreement + 0.5 * purity
        
    return confidences
