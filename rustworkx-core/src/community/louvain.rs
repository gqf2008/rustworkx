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

use hashbrown::HashMap;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};

use crate::community::random::fisher_yates_shuffle;
use crate::dictmap::{DictMap, InitWithHasher};

/// Louvain community detection algorithm.
///
/// This is a hierarchical, greedy, modularity-based community detection algorithm.
/// It works in two phases:
/// 1. **Local moving**: Each node is moved to the community of a neighbor if it
///    improves the modularity. This is repeated until no improvement is possible.
/// 2. **Aggregation**: Nodes in the same community are contracted into supernodes,
///    and the process repeats on the aggregated graph.
///
/// The algorithm terminates when no further modularity improvement is possible.
///
/// The algorithm is described in:
/// Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
/// Fast unfolding of communities in large networks. Journal of Statistical
/// Mechanics: Theory and Experiment, P10008.
///
/// # Arguments
///
/// * `graph` - The graph to analyze. Edge weights must be f64.
/// * `max_levels` - Maximum number of hierarchical levels to process. Default: 100.
/// * `max_pass_iterations` - Maximum number of inner local-moving iterations per level.
///   Default: 100.
/// * `resolution` - Resolution parameter (gamma). Values > 1 produce more communities,
///   values < 1 produce fewer. Default: 1.0.
/// * `seed` - Optional random seed for reproducibility. Default: 42.
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
/// use rustworkx_core::community::louvain_communities;
///
/// // Create a graph with two clear communities
/// let mut graph = UnGraph::<i32, f64>::new_undirected();
/// let a = graph.add_node(0);
/// let b = graph.add_node(1);
/// let c = graph.add_node(2);
/// let d = graph.add_node(3);
/// let e = graph.add_node(4);
/// let f = graph.add_node(5);
/// // Community 1: fully connected triangle
/// graph.add_edge(a, b, 1.0);
/// graph.add_edge(b, c, 1.0);
/// graph.add_edge(a, c, 1.0);
/// // Community 2: fully connected triangle
/// graph.add_edge(d, e, 1.0);
/// graph.add_edge(e, f, 1.0);
/// graph.add_edge(d, f, 1.0);
/// // Single bridge between communities
/// graph.add_edge(c, d, 1.0);
///
/// let communities = louvain_communities(&graph, None, None, None, None);
///
/// // Nodes in the same community should have the same label
/// assert_eq!(communities[&a], communities[&b]);
/// assert_eq!(communities[&b], communities[&c]);
/// assert_eq!(communities[&d], communities[&e]);
/// assert_eq!(communities[&e], communities[&f]);
/// assert_ne!(communities[&a], communities[&d]);
/// ```
pub fn louvain_communities<G>(
    graph: G,
    max_levels: Option<usize>,
    max_pass_iterations: Option<usize>,
    resolution: Option<f64>,
    seed: Option<u64>,
) -> DictMap<G::NodeId, u32>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoEdges + NodeCount,
    G::NodeId: std::cmp::Eq + std::hash::Hash + Copy + Send + Sync,
    G::EdgeWeight: Copy,
    f64: From<G::EdgeWeight>,
{
    let max_levels = max_levels.unwrap_or(100);
    let max_pass_iterations = max_pass_iterations.unwrap_or(100);
    let resolution = resolution.unwrap_or(1.0);
    let mut seed = seed.unwrap_or(42);

    let node_count = graph.node_count();
    if node_count == 0 {
        return DictMap::new();
    }

    // Collect nodes and map them to indices 0..n
    let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();
    let n = nodes.len();

    // Build adjacency list with weights and precompute self-loop weights
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut total_weight: f64 = 0.0;
    let mut self_loop: Vec<f64> = vec![0.0; n];

    for i in 0..n {
        for edge in graph.edges(nodes[i]) {
            let target_idx = graph.to_index(edge.target());
            let weight: f64 = f64::from(*edge.weight());
            adj[i].push((target_idx, weight));
            total_weight += weight;
            if target_idx == i {
                self_loop[i] += weight;
            }
        }
    }

    // For undirected graphs, petgraph's edges() returns each edge once,
    // but since we iterate all nodes, each undirected edge is counted twice.
    // m = total_weight / 2 = sum of all edge weights
    let m = total_weight / 2.0;

    if m == 0.0 {
        // No edges - each node is its own community
        let mut result = DictMap::with_capacity(n);
        for (i, &node) in nodes.iter().enumerate() {
            result.insert(node, i as u32);
        }
        return result;
    }

    // Compute node degrees (sum of edge weights, excluding self-loops)
    let mut node_degree: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for &(j, w) in &adj[i] {
            if j != i {
                node_degree[i] += w;
            }
        }
    }

    // Track mapping from original node to current aggregated-layer node.
    // Initially each original node is its own layer-0 node.
    let mut orig_to_layer: Vec<usize> = (0..n).collect();

    // Current graph data (starts as the original graph)
    let mut current_adj = adj;
    let mut current_degree = node_degree;
    let mut current_self_loop = self_loop;
    let mut current_m = m;
    let mut current_n = n;

    for _level in 0..max_levels {
        // Phase 1: Local moving on the current graph
        let mut layer_comm: Vec<u32> = (0..current_n).map(|i| i as u32).collect();
        louvain_pass(
            &mut layer_comm,
            &current_adj,
            &current_degree,
            &current_self_loop,
            current_m,
            resolution,
            max_pass_iterations,
            &mut seed,
        );

        // Check if any node moved from its initial singleton community
        let improved = layer_comm.iter().enumerate().any(|(i, &c)| c != i as u32);

        if !improved {
            break;
        }

        // Normalize community labels to a compact 0..k range for aggregation
        let (normalized_comm, _k) = normalize_communities(&layer_comm);

        // Map original nodes through this layer: compact community id becomes
        // the node index in the next aggregated graph.
        for i in 0..n {
            let comm = layer_comm[orig_to_layer[i]] as usize;
            orig_to_layer[i] = normalized_comm[comm] as usize;
        }

        // Phase 2: Aggregate communities into supernodes
        let next = aggregate_graph(&normalized_comm, &current_adj, current_n);
        if next.n == current_n {
            break; // No further aggregation possible
        }

        current_adj = next.adj;
        current_degree = next.degree;
        current_self_loop = next.self_loop;
        current_m = next.m;
        current_n = next.n;
    }

    // orig_to_layer now holds the final community for each original node.
    // Normalize to compact integers starting from 0.
    let final_labels: Vec<u32> = orig_to_layer.iter().map(|&x| x as u32).collect();
    crate::community::util::normalize_labels(&nodes, &final_labels)
}

/// Normalize community labels to a compact 0..k range.
fn normalize_communities(comm: &[u32]) -> (Vec<u32>, usize) {
    let mut label_map: HashMap<u32, u32> = HashMap::new();
    let mut next_label: u32 = 0;
    let mut normalized = Vec::with_capacity(comm.len());
    for &c in comm {
        let label = *label_map.entry(c).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        normalized.push(label);
    }
    (normalized, next_label as usize)
}

/// Aggregate a graph by contracting each community into a single supernode.
///
/// Each original edge (i, j, w) is mapped to (community[i], community[j], w).
/// Parallel edges are merged by summing weights. For undirected graphs,
/// adjacency lists contain each edge in both directions, so inter-community
/// edge weights are naturally double-counted — this is consistent with the
/// modularity formula which uses `2m` as the normalisation denominator.
struct AggregatedGraph {
    adj: Vec<Vec<(usize, f64)>>,
    degree: Vec<f64>,
    self_loop: Vec<f64>,
    m: f64,
    n: usize,
}

fn aggregate_graph(
    community: &[u32],
    adj: &[Vec<(usize, f64)>],
    n: usize,
) -> AggregatedGraph {
    let k = community.iter().copied().max().map(|c| c as usize + 1).unwrap_or(0);

    // Merge parallel edges using HashMap for O(deg) per node.
    // total_weight accumulates all directed edge weights; m = total_weight / 2.
    let mut temp_adj: Vec<HashMap<usize, f64>> = vec![HashMap::new(); k];
    let mut degree: Vec<f64> = vec![0.0; k];
    let mut total_weight: f64 = 0.0;

    for i in 0..n {
        let ci = community[i] as usize;
        for &(j, w) in &adj[i] {
            let cj = community[j] as usize;
            *temp_adj[ci].entry(cj).or_insert(0.0) += w;
            total_weight += w;
            if ci != cj {
                degree[ci] += w;
            }
        }
    }

    let mut new_adj: Vec<Vec<(usize, f64)>> = Vec::with_capacity(k);
    let mut self_loop: Vec<f64> = vec![0.0; k];
    for (ci, hm) in temp_adj.into_iter().enumerate() {
        self_loop[ci] = hm.get(&ci).copied().unwrap_or(0.0);
        new_adj.push(hm.into_iter().collect());
    }

    let m = total_weight / 2.0;

    AggregatedGraph {
        adj: new_adj,
        degree,
        self_loop,
        m,
        n: k,
    }
}

#[allow(clippy::too_many_arguments)]
fn louvain_pass(
    node_to_community: &mut [u32],
    adj: &[Vec<(usize, f64)>],
    node_degree: &[f64],
    self_loop: &[f64],
    m: f64,
    resolution: f64,
    max_pass_iterations: usize,
    seed: &mut u64,
) {
    let n = adj.len();
    let two_m = 2.0 * m;

    // Compute community statistics: sum of degrees per community
    let mut k_tot: HashMap<u32, f64> = HashMap::new();
    for i in 0..n {
        let c = node_to_community[i];
        *k_tot.entry(c).or_insert(0.0) += node_degree[i];
    }

    // Compute k_in: sum of edge weights within each community
    let mut k_in: HashMap<u32, f64> = HashMap::new();
    for i in 0..n {
        let ci = node_to_community[i];
        for &(j, w) in &adj[i] {
            if j != i && node_to_community[j] == ci {
                *k_in.entry(ci).or_insert(0.0) += w;
            }
        }
    }

    for _pass in 0..max_pass_iterations {
        let mut improved = false;

        // Shuffle order
        let mut order: Vec<usize> = (0..n).collect();
        fisher_yates_shuffle(&mut order, seed);

        for &i in &order {
            let ci = node_to_community[i];
            let ki = node_degree[i];
            if ki == 0.0 {
                continue;
            }

            // Remove node i from community ci
            let ki_in = edge_weight_to_community(i, ci, adj, node_to_community);
            let ki_self = self_loop[i];
            *k_in.entry(ci).or_insert(0.0) -= ki_in + ki_self;
            *k_tot.entry(ci).or_insert(0.0) -= ki;

            // Gather neighboring communities (include self-loop weight so
            // the node's current community is evaluated with its full attraction)
            let mut neighbor_comm: HashMap<u32, f64> = HashMap::new();
            for &(j, w) in &adj[i] {
                *neighbor_comm.entry(node_to_community[j]).or_insert(0.0) += w;
            }

            // Find best community
            let mut best_comm = ci;
            let mut best_dq = 0.0;

            for (&comm, &ki_to_comm) in &neighbor_comm {
                let ktot = *k_tot.get(&comm).unwrap_or(&0.0);
                // Modularity gain: ΔQ = ki_in/m - resolution * ktot * ki / (2m²)
                let dq = ki_to_comm / m - resolution * ktot * ki / (two_m * m);
                if dq > best_dq {
                    best_dq = dq;
                    best_comm = comm;
                }
            }

            // Move node
            node_to_community[i] = best_comm;
            let new_ki_in = edge_weight_to_community(i, best_comm, adj, node_to_community);
            let new_ki_self = self_loop[i];

            *k_in.entry(best_comm).or_insert(0.0) += new_ki_in + new_ki_self;
            *k_tot.entry(best_comm).or_insert(0.0) += ki;

            if best_comm != ci {
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }
}

/// Sum of edge weights from node i to nodes in community c (excluding node i itself)
fn edge_weight_to_community(
    node: usize,
    comm: u32,
    adj: &[Vec<(usize, f64)>],
    node_to_community: &[u32],
) -> f64 {
    let mut sum = 0.0;
    for &(j, w) in &adj[node] {
        if j != node && node_to_community[j] == comm {
            sum += w;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::graph::UnGraph;

    use super::louvain_communities;
    use crate::community::modularity::modularity;
    use crate::dictmap::{DictMap, InitWithHasher};

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

        let communities = louvain_communities(&graph, None, None, None, Some(42));

        // Same community nodes
        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        // Different communities
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_empty_graph() {
        let graph = UnGraph::<i32, f64>::new_undirected();
        let communities = louvain_communities(&graph, None, None, None, None);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let communities = louvain_communities(&graph, None, None, None, None);
        assert_eq!(communities.len(), 1);
        assert!(communities.contains_key(&a));
    }

    #[test]
    fn test_no_edges() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        for _ in 0..5 {
            graph.add_node(0);
        }
        let communities = louvain_communities(&graph, None, None, None, None);
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

        let communities = louvain_communities(&graph, None, None, None, Some(42));
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

        let communities = louvain_communities(&graph, None, None, None, Some(42));

        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_modularity() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);

        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(c, d, 1.0);

        let mut communities = DictMap::new();
        communities.insert(a, 0);
        communities.insert(b, 0);
        communities.insert(c, 1);
        communities.insert(d, 1);

        let q = modularity(&graph, &communities, None).unwrap();
        assert!(q > 0.0);
    }

    #[test]
    fn test_labels_normalized() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        graph.add_edge(a, b, 1.0);

        let communities = louvain_communities(&graph, None, None, None, None);
        // Labels should start from 0
        assert_eq!(communities[&a], 0);
        assert_eq!(communities[&b], 0);
    }

    #[test]
    fn test_seed_deterministic() {
        // Use a very simple graph where shuffle order is the only source of variation
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        // Path graph: a-b-c-d
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(c, d, 1.0);

        let c1 = louvain_communities(&graph, None, None, None, Some(123));
        let c2 = louvain_communities(&graph, None, None, None, Some(123));
        // Same number of communities
        let n1: HashSet<u32> = c1.values().copied().collect();
        let n2: HashSet<u32> = c2.values().copied().collect();
        assert_eq!(n1.len(), n2.len());
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
        graph.add_edge(c, d, -0.5); // negative edge

        let communities = louvain_communities(&graph, None, None, None, Some(42));
        // Should not panic; all nodes should be assigned
        assert_eq!(communities.len(), 4);
    }

    #[test]
    fn test_chain_of_cliques() {
        // Three cliques of 3 nodes each, connected by weak bridges.
        // This exercises the aggregation path with a larger graph.
        let mut graph = UnGraph::<i32, f64>::new_undirected();

        // Clique 0
        let c0: Vec<_> = (0..3).map(|_| graph.add_node(0)).collect();
        // Clique 1
        let c1: Vec<_> = (0..3).map(|_| graph.add_node(0)).collect();
        // Clique 2
        let c2: Vec<_> = (0..3).map(|_| graph.add_node(0)).collect();

        // Fully connect each clique
        for clique in [&c0, &c1, &c2] {
            for i in 0..clique.len() {
                for j in (i + 1)..clique.len() {
                    graph.add_edge(clique[i], clique[j], 10.0);
                }
            }
        }

        // Weak bridges between cliques
        graph.add_edge(c0[2], c1[0], 1.0);
        graph.add_edge(c1[2], c2[0], 1.0);

        let communities = louvain_communities(&graph, None, None, None, Some(42));
        assert_eq!(communities.len(), 9);

        // Each clique should be mostly within one community
        let comm_c0: HashSet<u32> = c0.iter().map(|n| communities[n]).collect();
        let comm_c1: HashSet<u32> = c1.iter().map(|n| communities[n]).collect();
        let comm_c2: HashSet<u32> = c2.iter().map(|n| communities[n]).collect();

        // With strong intra-clique edges, each clique should be in a single community
        assert_eq!(comm_c0.len(), 1, "Clique 0 nodes should be in same community");
        assert_eq!(comm_c1.len(), 1, "Clique 1 nodes should be in same community");
        assert_eq!(comm_c2.len(), 1, "Clique 2 nodes should be in same community");

        // The three cliques should be in distinct communities
        assert_ne!(comm_c0.iter().next().unwrap(), comm_c1.iter().next().unwrap());
        assert_ne!(comm_c1.iter().next().unwrap(), comm_c2.iter().next().unwrap());
    }
}
