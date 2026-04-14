// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

//! Walktrap community detection algorithm.
//!
//! This algorithm detects communities by using short random walks to compute
//! distances between nodes, then performing agglomerative hierarchical clustering.
//!
//! The algorithm is described in:
//! Pons, P., & Latapy, M. (2005). Computing communities in large networks
//! using random walks. Journal of Graph Algorithms and Applications, 10(2), 191-218.

use std::collections::BinaryHeap;

use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};

use crate::dictmap::{DictMap, InitWithHasher};

/// Wrapper for f64 that implements Ord for use in BinaryHeap (min-heap via Reverse).
#[derive(Debug, Clone, Copy, PartialEq)]
struct F64Ord(f64);

impl Eq for F64Ord {}

impl PartialOrd for F64Ord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F64Ord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Walktrap community detection algorithm.
///
/// This algorithm uses short random walks to measure similarity between nodes,
/// then performs agglomerative hierarchical clustering based on these distances.
/// The key insight is that random walks tend to stay within communities, so
/// nodes in the same community have similar random walk probability distributions.
///
/// # Arguments
///
/// * `graph` - The graph to analyze. Edge weights must be f64.
/// * `walk_length` - Length of random walks for computing distances.
///   Default: 4.
/// * `seed` - Optional random seed for reproducibility. Reserved for future
///   use; currently unused since the algorithm is deterministic.
///
/// # Returns
///
/// A `DictMap` mapping node index to community label (u32). Nodes in the same
/// community share the same label. Labels are normalized to compact integers
/// starting from 0.
///
/// # Example
///
/// ```rust
/// use petgraph::graph::UnGraph;
/// use rustworkx_core::community::walktrap_communities;
///
/// let mut graph = UnGraph::<i32, f64>::new_undirected();
/// let a = graph.add_node(0);
/// let b = graph.add_node(1);
/// let c = graph.add_node(2);
/// let d = graph.add_node(3);
/// let e = graph.add_node(4);
/// let f = graph.add_node(5);
/// // Community 1
/// graph.add_edge(a, b, 1.0);
/// graph.add_edge(b, c, 1.0);
/// graph.add_edge(a, c, 1.0);
/// // Community 2
/// graph.add_edge(d, e, 1.0);
/// graph.add_edge(e, f, 1.0);
/// graph.add_edge(d, f, 1.0);
/// // Bridge
/// graph.add_edge(c, d, 1.0);
///
/// let communities = walktrap_communities(&graph, None, None);
///
/// assert_eq!(communities[&a], communities[&b]);
/// assert_eq!(communities[&b], communities[&c]);
/// assert_eq!(communities[&d], communities[&e]);
/// assert_eq!(communities[&e], communities[&f]);
/// assert_ne!(communities[&a], communities[&d]);
/// ```
pub fn walktrap_communities<G>(
    graph: G,
    walk_length: Option<usize>,
    seed: Option<u64>,
) -> DictMap<G::NodeId, u32>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoEdges + NodeCount,
    G::NodeId: std::cmp::Eq + std::hash::Hash + Copy + Send + Sync,
    G::EdgeWeight: Copy,
    f64: From<G::EdgeWeight>,
{
    let t = walk_length.unwrap_or(4);
    let _seed = seed.unwrap_or(42);

    let node_count = graph.node_count();
    if node_count == 0 {
        return DictMap::new();
    }

    let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();
    let n = nodes.len();

    // Build adjacency list and compute degrees
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut degree: Vec<f64> = vec![0.0; n];

    for i in 0..n {
        for edge in graph.edges(nodes[i]) {
            let target_idx = graph.to_index(edge.target());
            let weight: f64 = f64::from(*edge.weight());
            adj[i].push((target_idx, weight));
            degree[i] += weight;
        }
    }

    // Compute transition matrix P: P[i][j] = A[i][j] / degree[i]
    // Compute P^t using repeated multiplication with double-buffered approach
    // Start with identity: prob[i][j] = 1 if i==j, 0 otherwise
    let mut prob: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        prob[i][i] = 1.0;
    }

    // Pre-allocate second buffer for swap (avoids O(n²) allocation per step)
    let mut new_prob: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    #[allow(clippy::needless_range_loop)]
    for _step in 0..t {
        // Zero out the new buffer
        for row in &mut new_prob {
            for v in row.iter_mut() {
                *v = 0.0;
            }
        }
        for i in 0..n {
            if degree[i] == 0.0 {
                new_prob[i].copy_from_slice(&prob[i]);
                continue;
            }
            for &(j, w) in &adj[i] {
                for k in 0..n {
                    new_prob[j][k] += prob[i][k] * w / degree[i];
                }
            }
        }
        std::mem::swap(&mut prob, &mut new_prob);
    }

    // Compute pairwise distances: d²(i,j) = Σ_k (degree[k]/deg_sum) * (prob[i][k] - prob[j][k])²
    let deg_sum: f64 = degree.iter().sum();
    let inv_deg_sum = if deg_sum > 0.0 { 1.0 / deg_sum } else { 0.0 };

    // Precompute sqrt(degree[k]/deg_sum) * prob[i][k] for each i,k
    let mut scaled: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in 0..n {
            scaled[i][k] = (degree[k] * inv_deg_sum).sqrt() * prob[i][k];
        }
    }

    // Compute upper triangle of distance matrix
    let mut dist: Vec<f64> = vec![0.0; n * (n - 1) / 2];
    for i in 0..n {
        for j in (i + 1)..n {
            let idx = i * n - i * (i + 1) / 2 + j - i - 1;
            let sum_sq: f64 = scaled[i]
                .iter()
                .zip(scaled[j].iter())
                .map(|(&si, &sj)| {
                    let diff = si - sj;
                    diff * diff
                })
                .sum();
            dist[idx] = sum_sq;
        }
    }

    // Agglomerative hierarchical clustering using Ward's criterion
    // Merge pairs with minimum distance and collect all intermediate partitions
    let (partitions, _merge_dists) = walktrap_cluster(n, &dist, &degree);

    // Select the partition with maximum modularity using a correct implementation
    // Q = (1/2m) * [Σ_{intra-community edges} w_ij - (1/2m) * Σ_{communities} (sum_deg_c)²]
    let two_m: f64 = degree.iter().sum(); // sum of all degrees = 2m
    if two_m == 0.0 {
        // No edges - return each node as its own community
        let mut result = DictMap::with_capacity(n);
        for (i, &node) in nodes.iter().enumerate() {
            result.insert(node, i as u32);
        }
        return result;
    }

    let mut best_mod = f64::NEG_INFINITY;
    let mut best_labels = partitions.last().cloned().unwrap_or_else(|| vec![0; n]);

    for labels in &partitions {
        // Compute modularity for this partition
        let mut sum_deg: Vec<f64> = vec![0.0; n]; // max possible community count
        for (node_idx, &comm) in labels.iter().enumerate() {
            sum_deg[comm as usize] += degree[node_idx];
        }
        let expected: f64 = sum_deg.iter().map(|d| d * d).sum::<f64>() / (2.0 * two_m);
        // Count intra-community edge weight
        let mut actual = 0.0;
        for i in 0..n {
            for edge in graph.edges(nodes[i]) {
                let j = graph.to_index(edge.target());
                let ci = labels[i];
                let cj = labels[j];
                if ci == cj {
                    actual += f64::from(*edge.weight());
                }
            }
        }
        actual /= 2.0; // each undirected edge counted twice
        let q = (actual - expected) / two_m * 2.0; // Q = (actual - expected) / m
        if q > best_mod {
            best_mod = q;
            best_labels = labels.clone();
        }
    }

    let mut result = DictMap::with_capacity(n);
    for (i, &node) in nodes.iter().enumerate() {
        result.insert(node, best_labels[i]);
    }
    result
}

/// Perform Ward's hierarchical clustering using a priority queue and return
/// all intermediate partitions along with the merge distances.
///
/// Uses a min-heap for O(n² log n) clustering instead of O(n³) linear scan.
/// Stale heap entries (from merged communities) are lazily skipped.
fn walktrap_cluster(
    n: usize,
    initial_dist: &[f64],
    degree: &[f64],
) -> (Vec<Vec<u32>>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![vec![0]], vec![]);
    }

    // Check if all pairwise distances are near zero (no structural information)
    let all_near_zero = initial_dist.iter().all(|&d| d < 1e-10);
    if all_near_zero {
        return (vec![(0..n as u32).collect()], vec![]);
    }

    // Initialize: each node is its own community
    let mut comm_nodes: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut comm_degree: Vec<f64> = degree.to_vec();
    let mut active: Vec<bool> = vec![true; n];
    let mut num_communities = n;

    // Distance matrix stored in flattened upper triangle
    let mut dist: Vec<f64> = initial_dist.to_vec();

    // Priority queue (min-heap via Reverse) storing (distance, ci, cj)
    let mut pq: BinaryHeap<(std::cmp::Reverse<F64Ord>, usize, usize)> =
        BinaryHeap::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist[idx(i, j, n)];
            pq.push((std::cmp::Reverse(F64Ord(d)), i, j));
        }
    }

    // Collect all partitions and merge distances
    let mut partitions: Vec<Vec<u32>> = Vec::new();
    let mut merge_dists: Vec<f64> = Vec::new();

    // Record initial partition (each node separate)
    partitions.push(assign_labels(&comm_nodes, &active, n));

    // Merge until one community remains
    while num_communities > 1 {
        // Pop minimum distance pair, skipping stale entries
        let (best_dist, ci, cj) = loop {
            let (std::cmp::Reverse(F64Ord(d)), a, b) = pq.pop().unwrap();
            if !active[a] || !active[b] {
                continue; // stale entry — one community already merged
            }
            break (d, a, b);
        };

        merge_dists.push(best_dist);

        // Merge cj into ci using Walktrap's Ward-like criterion
        let d_i = comm_degree[ci];
        let d_j = comm_degree[cj];
        let d_merged = d_i + d_j;

        // Merge node lists and degree sums
        let cj_nodes = comm_nodes[cj].clone();
        comm_nodes[ci].extend(cj_nodes);
        comm_degree[ci] = d_merged;

        // Update distances and push new entries for all active k
        let d_ij = dist[idx(ci, cj, n)];
        for k in (ci + 1)..n {
            if !active[k] {
                continue;
            }
            let d_ki = dist[idx(ci, k, n)];
            let d_kj = if k > cj {
                dist[idx(cj, k, n)]
            } else {
                // k > ci but k < cj, so pair is (k, cj)
                dist[idx(k, cj, n)]
            };
            let new_d = (d_i * d_ki + d_j * d_kj) / d_merged
                - d_i * d_j * d_ij / (d_merged * d_merged);
            let new_d = new_d.max(0.0);
            dist[idx(ci, k, n)] = new_d;
            pq.push((std::cmp::Reverse(F64Ord(new_d)), ci, k));
        }
        // For k < ci, also update (k, ci)
        for k in 0..ci {
            if !active[k] {
                continue;
            }
            let d_ki = dist[idx(k, ci, n)];
            let d_kj = dist[idx(k, cj, n)];
            let new_d = (d_i * d_ki + d_j * d_kj) / d_merged
                - d_i * d_j * d_ij / (d_merged * d_merged);
            let new_d = new_d.max(0.0);
            dist[idx(k, ci, n)] = new_d;
            pq.push((std::cmp::Reverse(F64Ord(new_d)), k, ci));
        }

        active[cj] = false;
        num_communities -= 1;

        // Record this partition
        partitions.push(assign_labels(&comm_nodes, &active, n));
    }

    (partitions, merge_dists)
}

fn idx(i: usize, j: usize, n: usize) -> usize {
    // Upper triangle index for pair (i,j) where i < j
    // Row i starts at position i*n - i*(i+1)/2
    // Within row i, column j is at offset (j - i - 1)
    i * n - i * (i + 1) / 2 + j - i - 1
}

fn assign_labels(comm_nodes: &[Vec<usize>], active: &[bool], n: usize) -> Vec<u32> {
    let mut labels = vec![0u32; n];
    let mut label_counter: u32 = 0;
    for ci in 0..comm_nodes.len() {
        if !active[ci] { continue; }
        for &node_idx in &comm_nodes[ci] {
            labels[node_idx] = label_counter;
        }
        label_counter += 1;
    }
    labels
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::graph::UnGraph;

    use super::walktrap_communities;

    #[test]
    fn test_two_communities() {
        // Two cliques of 5 nodes each, connected by a single bridge edge.
        // This gives clearer community structure that modularity can detect.
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        // Clique 1: nodes 0-4
        let c1: Vec<_> = (0..5).map(|_| graph.add_node(0)).collect();
        // Clique 2: nodes 5-9
        let c2: Vec<_> = (0..5).map(|_| graph.add_node(0)).collect();

        // All edges within clique 1
        for i in 0..c1.len() {
            for j in (i + 1)..c1.len() {
                graph.add_edge(c1[i], c1[j], 1.0);
            }
        }
        // All edges within clique 2
        for i in 0..c2.len() {
            for j in (i + 1)..c2.len() {
                graph.add_edge(c2[i], c2[j], 1.0);
            }
        }
        // Single bridge edge
        graph.add_edge(c1[0], c2[0], 1.0);

        let communities = walktrap_communities(&graph, None, None);
        assert_eq!(communities.len(), 10);

        // All nodes in clique 1 should be in same community
        for i in 1..c1.len() {
            assert_eq!(communities[&c1[0]], communities[&c1[i]],
                "Nodes {:?} and {:?} should be in same community", c1[0], c1[i]);
        }
        // All nodes in clique 2 should be in same community
        for i in 1..c2.len() {
            assert_eq!(communities[&c2[0]], communities[&c2[i]],
                "Nodes {:?} and {:?} should be in same community", c2[0], c2[i]);
        }
        // The two cliques should be in different communities
        assert_ne!(communities[&c1[0]], communities[&c2[0]],
            "The two cliques should be in different communities");
    }

    #[test]
    fn test_empty_graph() {
        let graph = UnGraph::<i32, f64>::new_undirected();
        let communities = walktrap_communities(&graph, None, None);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let communities = walktrap_communities(&graph, None, None);
        assert_eq!(communities.len(), 1);
        assert!(communities.contains_key(&a));
    }

    #[test]
    fn test_no_edges() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        for _ in 0..5 {
            graph.add_node(0);
        }
        let communities = walktrap_communities(&graph, None, None);
        assert_eq!(communities.len(), 5);
        let labels: Vec<u32> = communities.values().copied().collect();
        let unique: HashSet<u32> = labels.into_iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_complete_graph_single_community() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let nodes: Vec<_> = (0..10).map(|_| graph.add_node(0)).collect();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                graph.add_edge(nodes[i], nodes[j], 1.0);
            }
        }

        let communities = walktrap_communities(&graph, None, None);
        let labels: Vec<u32> = communities.values().copied().collect();
        let unique: HashSet<u32> = labels.into_iter().collect();
        assert_eq!(unique.len(), 1);
    }

    #[test]
    fn test_weighted_edges() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);
        let f = graph.add_node(5);

        graph.add_edge(a, b, 10.0);
        graph.add_edge(b, c, 10.0);
        graph.add_edge(a, c, 10.0);
        graph.add_edge(d, e, 10.0);
        graph.add_edge(e, f, 10.0);
        graph.add_edge(d, f, 10.0);
        graph.add_edge(c, d, 0.1);

        let communities = walktrap_communities(&graph, None, None);

        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_walk_length_parameter() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(c, d, 1.0);

        // Different walk lengths should produce valid results
        let short = walktrap_communities(&graph, Some(2), None);
        let long = walktrap_communities(&graph, Some(10), None);
        assert_eq!(short.len(), 4);
        assert_eq!(long.len(), 4);
    }

    #[test]
    fn test_labels_normalized() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        graph.add_edge(a, b, 1.0);

        let communities = walktrap_communities(&graph, None, None);
        assert_eq!(communities[&a], 0);
        assert_eq!(communities[&b], 0);
    }
}
