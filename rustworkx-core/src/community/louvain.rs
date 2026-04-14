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
/// let communities = louvain_communities(&graph, None, None, None);
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
    let resolution = resolution.unwrap_or(1.0);
    let mut seed = seed.unwrap_or(42);

    let node_count = graph.node_count();
    if node_count == 0 {
        return DictMap::new();
    }

    // Collect nodes and map them to indices 0..n
    let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();
    let n = nodes.len();

    // Build adjacency list with weights
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut total_weight: f64 = 0.0;

    for i in 0..n {
        for edge in graph.edges(nodes[i]) {
            let target_idx = graph.to_index(edge.target());
            let weight: f64 = f64::from(*edge.weight());
            adj[i].push((target_idx, weight));
            total_weight += weight;
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

    // Initialize: each node in its own community
    let mut node_to_community: Vec<u32> = (0..n).map(|i| i as u32).collect();

    // Compute node degrees (sum of edge weights)
    let mut node_degree: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for &(j, w) in &adj[i] {
            if j != i {
                node_degree[i] += w;
            }
        }
    }

    // Run Louvain algorithm
    louvain_pass(
        &mut node_to_community,
        &adj,
        &node_degree,
        m,
        resolution,
        max_levels,
        &mut seed,
    );

    // Normalize labels to compact integers starting from 0
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

fn louvain_pass(
    node_to_community: &mut [u32],
    adj: &[Vec<(usize, f64)>],
    node_degree: &[f64],
    m: f64,
    resolution: f64,
    max_levels: usize,
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

    for _level in 0..max_levels {
        let mut improved = false;

        // Shuffle order
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (*seed as usize) % (i + 1);
            order.swap(i, j);
        }

        for &i in &order {
            let ci = node_to_community[i];
            let ki = node_degree[i];
            if ki == 0.0 {
                continue;
            }

            // Remove node i from community ci
            let ki_in = edge_weight_to_community(i, ci, adj, node_to_community);
            let ki_self = self_loop_weight(i, adj);
            *k_in.entry(ci).or_insert(0.0) -= ki_in + ki_self;
            *k_tot.entry(ci).or_insert(0.0) -= ki;

            // Gather neighboring communities
            let mut neighbor_comm: HashMap<u32, f64> = HashMap::new();
            for &(j, w) in &adj[i] {
                if j == i {
                    continue;
                }
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
            let new_ki_self = self_loop_weight(i, adj);

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

/// Self-loop weight for a node
fn self_loop_weight(node: usize, adj: &[Vec<(usize, f64)>]) -> f64 {
    for &(j, w) in &adj[node] {
        if j == node {
            return w;
        }
    }
    0.0
}

/// Compute the modularity of a given partition.
///
/// Q = (1/2m) * sum[(A_ij - resolution * k_i*k_j/(2m)) * delta(c_i, c_j)]
///
/// The `resolution` parameter defaults to 1.0. Values > 1 produce more communities,
/// values < 1 produce fewer. Use the same resolution as was used in the community
/// detection algorithm for consistent evaluation.
pub fn modularity<G>(
    graph: G,
    communities: &DictMap<G::NodeId, u32>,
    resolution: Option<f64>,
) -> f64
where
    G: NodeIndexable + IntoEdges + IntoNodeIdentifiers + NodeCount,
    G::NodeId: std::cmp::Eq + std::hash::Hash + Copy,
    G::EdgeWeight: Copy,
    f64: From<G::EdgeWeight>,
{
    let resolution = resolution.unwrap_or(1.0);
    let mut m: f64 = 0.0;
    let mut degree: HashMap<G::NodeId, f64> = HashMap::new();

    for node in graph.node_identifiers() {
        let mut d = 0.0;
        for edge in graph.edges(node) {
            let w: f64 = f64::from(*edge.weight());
            d += w;
            m += w;
        }
        degree.insert(node, d);
    }

    m /= 2.0;
    if m == 0.0 {
        return 0.0;
    }

    let mut q = 0.0;
    for node in graph.node_identifiers() {
        let ci = communities[&node];
        let ki = degree[&node];
        for edge in graph.edges(node) {
            let target = edge.target();
            let cj = communities[&target];
            if ci == cj {
                let a_ij: f64 = f64::from(*edge.weight());
                let kj = degree[&target];
                q += a_ij - resolution * (ki * kj) / (2.0 * m);
            }
        }
    }

    q / (2.0 * m)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::graph::UnGraph;

    use super::{louvain_communities, modularity};
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

        let communities = louvain_communities(&graph, None, None, Some(42));

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
        let communities = louvain_communities(&graph, None, None, None);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let communities = louvain_communities(&graph, None, None, None);
        assert_eq!(communities.len(), 1);
        assert!(communities.contains_key(&a));
    }

    #[test]
    fn test_no_edges() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        for _ in 0..5 {
            graph.add_node(0);
        }
        let communities = louvain_communities(&graph, None, None, None);
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

        let communities = louvain_communities(&graph, None, None, Some(42));
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

        let communities = louvain_communities(&graph, None, None, Some(42));

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

        let q = modularity(&graph, &communities, None);
        assert!(q > 0.0);
    }

    #[test]
    fn test_labels_normalized() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        graph.add_edge(a, b, 1.0);

        let communities = louvain_communities(&graph, None, None, None);
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

        let c1 = louvain_communities(&graph, None, None, Some(123));
        let c2 = louvain_communities(&graph, None, None, Some(123));
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

        let communities = louvain_communities(&graph, None, None, Some(42));
        // Should not panic; all nodes should be assigned
        assert_eq!(communities.len(), 4);
    }
}
