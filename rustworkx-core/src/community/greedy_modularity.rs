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

//! Greedy modularity optimization community detection algorithm.
//!
//! This algorithm starts with each node in its own community and
//! iteratively merges the pair of communities that produces the
//! largest increase in modularity, until no further improvement
//! is possible.
//!
//! The algorithm is described in:
//! Clauset, A., Newman, M. E. J., & Moore, C. (2004).
//! Finding community structure in very large networks.
//! Physical Review E, 70(6), 066111.

use hashbrown::{HashMap, HashSet};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use priority_queue::PriorityQueue;

use crate::dictmap::{DictMap, InitWithHasher};

/// Wrapper for f64 that implements Ord for use in PriorityQueue.
#[derive(Debug, Clone, Copy)]
struct F64Ord(f64);

impl PartialEq for F64Ord {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
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

/// Greedy modularity optimization community detection algorithm.
///
/// This is a hierarchical, agglomerative community detection algorithm
/// that starts with each node in its own community and merges the pair
/// of communities that maximizes modularity at each step.
///
/// # Arguments
///
/// * `graph` - The graph to analyze. Edge weights must be f64.
/// * `resolution` - Resolution parameter (gamma). Values > 1 produce more
///   communities, values < 1 produce fewer. Default: 1.0.
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
/// use rustworkx_core::community::greedy_modularity_communities;
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
/// let communities = greedy_modularity_communities(&graph, None);
///
/// assert_eq!(communities[&a], communities[&b]);
/// assert_eq!(communities[&b], communities[&c]);
/// assert_eq!(communities[&d], communities[&e]);
/// assert_eq!(communities[&e], communities[&f]);
/// assert_ne!(communities[&a], communities[&d]);
/// ```
pub fn greedy_modularity_communities<G>(
    graph: G,
    resolution: Option<f64>,
) -> DictMap<G::NodeId, u32>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoEdges + NodeCount,
    G::NodeId: std::cmp::Eq + std::hash::Hash + Copy + Send + Sync,
    G::EdgeWeight: Copy,
    f64: From<G::EdgeWeight>,
{
    let resolution = resolution.unwrap_or(1.0);
    let node_count = graph.node_count();
    if node_count == 0 {
        return DictMap::new();
    }

    let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();
    let n = nodes.len();

    // Build adjacency list with weights and compute total weight
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut total_weight: f64 = 0.0;
    let mut degree: Vec<f64> = vec![0.0; n];

    for i in 0..n {
        for edge in graph.edges(nodes[i]) {
            let target_idx = graph.to_index(edge.target());
            let weight: f64 = f64::from(*edge.weight());
            adj[i].push((target_idx, weight));
            // For undirected graphs, total_weight counts each edge once.
            // We accumulate all directed edge weights and divide by 2.
            total_weight += weight;
            // Degree excluding self-loops
            if target_idx != i {
                degree[i] += weight;
            }
        }
    }

    let m = total_weight / 2.0;

    if m == 0.0 {
        let mut result = DictMap::with_capacity(n);
        for (i, &node) in nodes.iter().enumerate() {
            result.insert(node, i as u32);
        }
        return result;
    }

    let two_m = 2.0 * m;

    // Initialize: each node in its own community
    let mut node_to_community: Vec<u32> = (0..n).map(|i| i as u32).collect();

    // a[i] = degree of community i / (2m)
    let mut a: Vec<f64> = degree.iter().map(|&d| d / two_m).collect();

    // e[i][j] = sum of edge weights between communities i and j / (2m)
    // Stored as Vec of HashMaps for O(deg) neighbor lookup
    let mut e: Vec<HashMap<u32, f64>> = (0..n).map(|_| HashMap::new()).collect();

    // Build initial e matrix
    for i in 0..n {
        for &(j, w) in &adj[i] {
            let ci = node_to_community[i];
            let cj = node_to_community[j];
            if ci != cj {
                *e[ci as usize].entry(cj).or_insert(0.0) += w / two_m;
                *e[cj as usize].entry(ci).or_insert(0.0) += w / two_m;
            }
        }
    }

    // Priority queue of (delta_Q, (ci, cj)) for all connected community pairs
    // delta_Q = 2 * (e[i][j] - resolution * a[i] * a[j])
    let mut queue: PriorityQueue<(u32, u32), F64Ord> = PriorityQueue::new();

    for ci in 0..n {
        for (&cj, &eij) in &e[ci] {
            if (ci as u32) < cj {
                let dq = 2.0 * (eij - resolution * a[ci] * a[cj as usize]);
                queue.push((ci as u32, cj), F64Ord(dq));
            }
        }
    }

    // Track which communities still exist (haven't been merged into another)
    let mut active: Vec<bool> = vec![true; n];

    // Track nodes per community for relabeling
    let mut comm_nodes: HashMap<u32, Vec<usize>> = HashMap::new();
    for i in 0..n {
        comm_nodes.entry(i as u32).or_default().push(i);
    }

    // Greedy merging
    while let Some(((ci, cj), F64Ord(dq))) = queue.pop() {
        // Stop if no improvement
        if dq <= 0.0 {
            break;
        }

        // Skip if either community is no longer active
        if !active[ci as usize] || !active[cj as usize] {
            continue;
        }

        // Merge cj into ci
        active[cj as usize] = false;
        a[ci as usize] += a[cj as usize];
        a[cj as usize] = 0.0;

        // Update node assignments
        if let Some(nodes_in_cj) = comm_nodes.remove(&cj) {
            for &node_idx in &nodes_in_cj {
                node_to_community[node_idx] = ci;
            }
            comm_nodes.entry(ci).or_default().extend(nodes_in_cj);
        }

        // Update e values and queue — only iterate neighbors, not all entries
        // Collect neighbors of ci and cj via direct HashMap lookups: O(deg(ci) + deg(cj))
        let mut neighbors_to_update: HashSet<u32> = HashSet::new();
        for (&ck, _) in &e[ci as usize] {
            if active[ck as usize] && ck != cj {
                neighbors_to_update.insert(ck);
            }
        }
        for (&ck, _) in &e[cj as usize] {
            if active[ck as usize] && ck != ci {
                neighbors_to_update.insert(ck);
            }
        }

        for &ck in &neighbors_to_update {
            if ck == ci || !active[ck as usize] {
                continue;
            }

            let eik = *e[ci as usize].get(&ck).unwrap_or(&0.0);
            let ejk = *e[cj as usize].get(&ck).unwrap_or(&0.0);

            // New combined edge weight
            let e_new = eik + ejk;
            if e_new > 0.0 {
                // Remove old entries from both sides
                e[ck as usize].remove(&ci);
                e[ck as usize].remove(&cj);
                e[ci as usize].remove(&ck);
                e[cj as usize].remove(&ck);
                // Insert new entries
                e[ci as usize].insert(ck, e_new);
                e[ck as usize].insert(ci, e_new);
                let dq_new = 2.0 * (e_new - resolution * a[ci as usize] * a[ck as usize]);
                queue.push((ci, ck), F64Ord(dq_new));
            } else {
                // No edge weight - just remove entries
                e[ck as usize].remove(&ci);
                e[ck as usize].remove(&cj);
                e[ci as usize].remove(&ck);
                e[cj as usize].remove(&ck);
            }
        }

        // All active neighbors of cj were already processed in neighbors_to_update.
        // Clear merged community's entries.
        e[cj as usize].clear();
    }

    // Normalize labels to compact integers starting from 0
    crate::community::util::normalize_labels(&nodes, &node_to_community)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::graph::UnGraph;

    use super::greedy_modularity_communities;

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

        let communities = greedy_modularity_communities(&graph, None);

        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_empty_graph() {
        let graph = UnGraph::<i32, f64>::new_undirected();
        let communities = greedy_modularity_communities(&graph, None);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let communities = greedy_modularity_communities(&graph, None);
        assert_eq!(communities.len(), 1);
        assert!(communities.contains_key(&a));
    }

    #[test]
    fn test_no_edges() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        for _ in 0..5 {
            graph.add_node(0);
        }
        let communities = greedy_modularity_communities(&graph, None);
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

        let communities = greedy_modularity_communities(&graph, None);
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

        let communities = greedy_modularity_communities(&graph, None);

        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_resolution_parameter() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(c, d, 1.0);

        // High resolution should produce more communities
        let hi_res = greedy_modularity_communities(&graph, Some(2.0));
        let lo_res = greedy_modularity_communities(&graph, Some(0.5));

        let hi_unique: HashSet<u32> = hi_res.values().copied().collect();
        let lo_unique: HashSet<u32> = lo_res.values().copied().collect();

        // Higher resolution -> more communities (or equal)
        assert!(hi_unique.len() >= lo_unique.len());
    }

    #[test]
    fn test_labels_normalized() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        graph.add_edge(a, b, 1.0);

        let communities = greedy_modularity_communities(&graph, None);
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

        let communities = greedy_modularity_communities(&graph, None);
        assert_eq!(communities.len(), 4);
    }

    #[test]
    fn test_self_loops() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(a, a, 5.0);

        let communities = greedy_modularity_communities(&graph, None);
        assert_eq!(communities.len(), 3);
    }
}
