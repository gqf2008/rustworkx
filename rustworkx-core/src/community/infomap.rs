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

//! Infomap community detection algorithm.
//!
//! This algorithm detects communities by minimizing the map equation,
//! an information-theoretic measure for the description length of
//! random walks on the graph.
//!
//! The algorithm is described in:
//! Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks
//! on complex networks reveal community structure.
//! Proceedings of the National Academy of Sciences, 105(4), 1118-1123.

use hashbrown::HashMap;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};

use crate::dictmap::{DictMap, InitWithHasher};

/// Infomap community detection algorithm.
///
/// This algorithm detects communities by minimizing the map equation,
/// an information-theoretic objective that finds the partition that
/// best compresses the description of random walks on the graph.
///
/// # Arguments
///
/// * `graph` - The graph to analyze. Edge weights must be f64.
/// * `max_iterations` - Maximum number of iterations for the optimization.
///   Default: 100.
/// * `teleport_prob` - Teleportation probability for the random walk
///   (PageRank-style damping). Default: 0.15.
/// * `seed` - Optional random seed for reproducibility.
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
/// use rustworkx_core::community::infomap_communities;
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
/// let communities = infomap_communities(&graph, None, None, None);
///
/// assert_eq!(communities[&a], communities[&b]);
/// assert_eq!(communities[&b], communities[&c]);
/// assert_eq!(communities[&d], communities[&e]);
/// assert_eq!(communities[&e], communities[&f]);
/// assert_ne!(communities[&a], communities[&d]);
/// ```
pub fn infomap_communities<G>(
    graph: G,
    max_iterations: Option<usize>,
    teleport_prob: Option<f64>,
    seed: Option<u64>,
) -> DictMap<G::NodeId, u32>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoEdges + NodeCount,
    G::NodeId: std::cmp::Eq + std::hash::Hash + Copy + Send + Sync,
    G::EdgeWeight: Copy,
    f64: From<G::EdgeWeight>,
{
    let max_iterations = max_iterations.unwrap_or(100);
    let teleport = teleport_prob.unwrap_or(0.15);
    let mut seed = seed.unwrap_or(42);

    let node_count = graph.node_count();
    if node_count == 0 {
        return DictMap::new();
    }

    let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();
    let n = nodes.len();

    // Build adjacency list with weights
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut out_weight: Vec<f64> = vec![0.0; n];

    for i in 0..n {
        for edge in graph.edges(nodes[i]) {
            let target_idx = graph.to_index(edge.target());
            let weight: f64 = f64::from(*edge.weight());
            adj[i].push((target_idx, weight));
            out_weight[i] += weight;
        }
    }

    // Compute stationary distribution via power iteration (double-buffered)
    let visit_rate =
        compute_stationary_distribution(&adj, &out_weight, n, teleport);

    // Build reverse adjacency list for sparse incoming flow lookup.
    let mut rev_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (i, neighbors) in adj.iter().enumerate() {
        for &(j, w) in neighbors {
            rev_adj[j].push((i, w));
        }
    }

    // Initialize: each node in its own community
    let mut node_to_community: Vec<u32> = (0..n).map(|i| i as u32).collect();

    // Reusable mutual flow buffer — avoids O(n) allocation per node per iteration
    let mut mutual_flow: HashMap<usize, f64> = HashMap::new();

    // Shuffle order
    let mut order: Vec<usize> = (0..n).collect();

    for _iteration in 0..max_iterations {
        // Shuffle using LCG
        for i in (1..n).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed as usize) % (i + 1);
            order.swap(i, j);
        }

        let mut improved = false;
        let teleport_per_node = teleport / n as f64;

        for &i in &order {
            if out_weight[i] == 0.0 {
                continue;
            }

            let ci = node_to_community[i] as usize;

            // Compute mutual flow between node i and each community.
            // mutual_flow[c] = Σ_{j∈c, j≠i} (flow(i→j) + flow(j→i))
            // flow(i→j) = visit_rate[i] * P(i→j)
            // P(i→j) = (1-teleport) * w/out_weight[i] + teleport/n
            //
            // We compute this sparsely:
            // 1. Outgoing neighbors via adj[i]  — O(deg_out(i))
            // 2. Incoming neighbors via rev_adj[i] — O(deg_in(i))
            // Total per-node cost: O(deg(i)) instead of O(n).

            mutual_flow.clear();

            // Outgoing flow: i -> j (sparse via adjacency list)
            for &(j, w) in &adj[i] {
                if j == i {
                    continue;
                }
                let cj = node_to_community[j] as usize;
                let p_ij = (1.0 - teleport) * w / out_weight[i] + teleport_per_node;
                *mutual_flow.entry(cj).or_insert(0.0) += visit_rate[i] * p_ij;
            }

            // Incoming flow: j -> i (sparse via reverse adjacency list)
            for &(j, w) in &rev_adj[i] {
                if j == i {
                    continue;
                }
                let cj = node_to_community[j] as usize;
                let p_ji = (1.0 - teleport) * w / out_weight[j] + teleport_per_node;
                *mutual_flow.entry(cj).or_insert(0.0) += visit_rate[j] * p_ji;
            }

            // Find best community (highest mutual flow)
            let mut best_comm = ci;
            let mut best_flow_val = *mutual_flow.get(&ci).unwrap_or(&0.0);

            for (&cj, &mf) in &mutual_flow {
                if cj == ci {
                    continue;
                }
                if mf > best_flow_val {
                    best_flow_val = mf;
                    best_comm = cj;
                }
            }

            // Move if improvement
            if best_comm != ci {
                node_to_community[i] = best_comm as u32;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    // Normalize labels
    let mut label_map: HashMap<u32, u32> = HashMap::new();
    let mut next_label: u32 = 0;

    let mut result = DictMap::with_capacity(n);
    for (i, &node) in nodes.iter().enumerate() {
        let raw_label = node_to_community[i];
        let compact_label = label_map.entry(raw_label).or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });
        result.insert(node, *compact_label);
    }
    result
}

/// Compute stationary distribution
/// Uses double-buffered approach (swap instead of clone).
fn compute_stationary_distribution(
    adj: &[Vec<(usize, f64)>],
    out_weight: &[f64],
    n: usize,
    teleport: f64,
) -> Vec<f64> {
    let mut p: Vec<f64> = vec![1.0 / n as f64; n];
    let mut p_new: Vec<f64> = vec![0.0; n];
    let tol = 1e-9;
    let max_iter = 1000;

    for _ in 0..max_iter {
        // Zero out the new buffer
        for v in p_new.iter_mut().take(n) {
            *v = 0.0;
        }

        // Distribute probability mass via random walk transitions
        let teleport_share = teleport / n as f64;
        let one_minus_t = 1.0 - teleport;
        for i in 0..n {
            let pi = p[i];
            if out_weight[i] > 0.0 {
                let scale = one_minus_t * pi;
                for &(j, w) in &adj[i] {
                    p_new[j] += scale * w / out_weight[i];
                }
            }
        }
        // Teleport: distribute uniformly from all nodes (O(n) instead of O(n²))
        let total_teleport: f64 = p.iter().sum::<f64>() * teleport_share;
        for v in p_new.iter_mut().take(n) {
            *v += total_teleport;
        }

        // Normalize
        let sum: f64 = p_new.iter().sum();
        if sum > 0.0 {
            for v in &mut p_new {
                *v /= sum;
            }
        }

        // Check convergence
        let diff: f64 = (0..n).map(|k| (p_new[k] - p[k]).abs()).sum();
        // Swap buffers instead of cloning
        std::mem::swap(&mut p, &mut p_new);
        if diff < tol {
            break;
        }
    }

    p
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::graph::UnGraph;

    use super::infomap_communities;

    #[test]
    fn test_two_communities() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);
        let f = graph.add_node(5);

        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(a, c, 1.0);
        graph.add_edge(d, e, 1.0);
        graph.add_edge(e, f, 1.0);
        graph.add_edge(d, f, 1.0);
        graph.add_edge(c, d, 1.0);

        let communities = infomap_communities(&graph, None, None, Some(42));

        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_empty_graph() {
        let graph = UnGraph::<i32, f64>::new_undirected();
        let communities = infomap_communities(&graph, None, None, None);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let communities = infomap_communities(&graph, None, None, None);
        assert_eq!(communities.len(), 1);
        assert!(communities.contains_key(&a));
    }

    #[test]
    fn test_no_edges() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        for _ in 0..5 {
            graph.add_node(0);
        }
        let communities = infomap_communities(&graph, None, None, None);
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

        let communities = infomap_communities(&graph, None, None, Some(42));
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

        let communities = infomap_communities(&graph, None, None, Some(42));

        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_seed_deterministic() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(c, d, 1.0);

        let c1 = infomap_communities(&graph, None, None, Some(123));
        let c2 = infomap_communities(&graph, None, None, Some(123));

        let n1: HashSet<u32> = c1.values().copied().collect();
        let n2: HashSet<u32> = c2.values().copied().collect();
        assert_eq!(n1.len(), n2.len());
    }

    #[test]
    fn test_labels_normalized() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        graph.add_edge(a, b, 1.0);

        let communities = infomap_communities(&graph, None, None, None);
        assert_eq!(communities[&a], 0);
        assert_eq!(communities[&b], 0);
    }

    #[test]
    fn test_negative_weights() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(c, d, -0.5);

        let communities = infomap_communities(&graph, None, None, Some(42));
        assert_eq!(communities.len(), 4);
    }
}
