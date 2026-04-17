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
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::dictmap::{DictMap, InitWithHasher};

/// Label propagation community detection algorithm.
///
/// This is a fast community detection algorithm that works by having each node
/// adopt the label that is most frequent among its neighbors. The algorithm
/// iterates until labels stop changing or the maximum number of iterations
/// is reached.
///
/// The algorithm is described in:
/// Raghavan, U. N., Albert, R., & Kumara, S. (2007). Near linear time algorithm
/// to detect community structures in large-scale networks. Physical review E,
/// 76(3), 036106.
///
/// # Arguments
///
/// * `graph` - The graph to analyze
/// * `max_iterations` - Maximum number of iterations to perform
/// * `seed` - Optional random seed for reproducibility. When not provided,
///   results may vary between runs due to random tie-breaking and node ordering.
///
/// # Returns
///
/// A `DictMap` mapping node index to community label (u32). Nodes in the same
/// community share the same label.
///
/// # Example
///
/// ```rust
/// use petgraph::graph::UnGraph;
/// use rustworkx_core::community::label_propagation;
///
/// // Create a graph with two clear communities
/// let mut graph = UnGraph::<i32, i32>::new_undirected();
/// let a = graph.add_node(0);
/// let b = graph.add_node(1);
/// let c = graph.add_node(2);
/// let d = graph.add_node(3);
/// let e = graph.add_node(4);
/// let f = graph.add_node(5);
/// // Community 1: a, b, c
/// graph.add_edge(a, b, 1);
/// graph.add_edge(b, c, 1);
/// graph.add_edge(a, c, 1);
/// // Community 2: d, e, f
/// graph.add_edge(d, e, 1);
/// graph.add_edge(e, f, 1);
/// graph.add_edge(d, f, 1);
/// // Bridge between communities
/// graph.add_edge(c, d, 1);
///
/// let communities = label_propagation(&graph, 100, Some(42));
///
/// // Nodes in the same community should have the same label
/// assert_eq!(communities[&a], communities[&b]);
/// assert_eq!(communities[&b], communities[&c]);
/// assert_eq!(communities[&d], communities[&e]);
/// assert_eq!(communities[&e], communities[&f]);
/// // The two communities should have different labels
/// assert_ne!(communities[&a], communities[&d]);
/// ```
pub fn label_propagation<G>(
    graph: G,
    max_iterations: usize,
    seed: Option<u64>,
) -> DictMap<G::NodeId, u32>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoNeighbors + NodeCount + Send + Sync,
    G::NodeId: std::cmp::Eq + std::hash::Hash + Copy + Send + Sync,
{
    let node_count = graph.node_count();
    if node_count == 0 {
        return DictMap::new();
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            rand::rng().fill_bytes(&mut seed_bytes);
            StdRng::from_seed(seed_bytes)
        }
    };

    // Collect all nodes into a vector for indexed access
    let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();
    let n = nodes.len();

    // Initialize: each node gets its own unique label
    let mut labels: Vec<u32> = (0..n as u32).collect();

    // Build adjacency list using node indices (0..n)
    let adj: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let graph_ref = &graph;
            graph_ref
                .neighbors(nodes[i])
                .map(|neighbor| graph_ref.to_index(neighbor))
                .collect()
        })
        .collect();

    // Create a shuffled order for iteration
    let mut node_order: Vec<usize> = (0..n).collect();

    for _iteration in 0..max_iterations {
        // Shuffle node order each iteration
        node_order.shuffle(&mut rng);

        let mut changed = false;

        // Process nodes in random order
        for &node_idx in &node_order {
            let neighbors = &adj[node_idx];
            if neighbors.is_empty() {
                continue;
            }

            // Count label frequencies among neighbors
            let mut label_counts: HashMap<u32, usize> = HashMap::new();
            for &neighbor_idx in neighbors {
                *label_counts.entry(labels[neighbor_idx]).or_insert(0) += 1;
            }

            // Find the maximum frequency
            let max_freq = label_counts.values().copied().max().unwrap();

            // Collect all labels with maximum frequency (for tie-breaking)
            let candidates: Vec<u32> = label_counts
                .into_iter()
                .filter(|(_, count)| *count == max_freq)
                .map(|(label, _)| label)
                .collect();

            // Random tie-breaking
            let new_label = if candidates.len() == 1 {
                candidates[0]
            } else {
                *candidates.choose(&mut rng).unwrap()
            };

            if new_label != labels[node_idx] {
                labels[node_idx] = new_label;
                changed = true;
            }
        }

        // Early termination if no changes
        if !changed {
            break;
        }
    }

    // Convert back to NodeId keys with normalized community labels (0, 1, 2, ...)
    let mut label_map: HashMap<u32, u32> = HashMap::new();
    let mut next_label: u32 = 0;

    let mut result = DictMap::with_capacity(n);
    for (i, &node) in nodes.iter().enumerate() {
        let raw_label = labels[i];
        let compact_label = label_map.entry(raw_label).or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });
        result.insert(node, *compact_label);
    }
    result
}

#[cfg(test)]
mod tests {
    use petgraph::graph::{DiGraph, UnGraph};

    use crate::dictmap::DictMap;

    use super::label_propagation;

    #[test]
    fn test_two_communities() {
        let mut graph = UnGraph::<i32, i32>::new_undirected();
        // Community 1: 0, 1, 2 (fully connected)
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        // Community 2: 3, 4, 5 (fully connected)
        let d = graph.add_node(3);
        let e = graph.add_node(4);
        let f = graph.add_node(5);

        graph.add_edge(a, b, 1);
        graph.add_edge(b, c, 1);
        graph.add_edge(a, c, 1);
        graph.add_edge(d, e, 1);
        graph.add_edge(e, f, 1);
        graph.add_edge(d, f, 1);
        // Single bridge edge
        graph.add_edge(c, d, 1);

        let communities = label_propagation(&graph, 100, Some(42));

        // Verify all nodes are present and labels are normalized (compact)
        assert_eq!(communities.len(), 6);
        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        // Labels should be compact (0, 1, 2, ...)
        let unique_labels: Vec<_> = communities
            .values()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let max_label = unique_labels.iter().max().copied().unwrap_or(0);
        assert!((max_label as usize) < unique_labels.len());
    }

    #[test]
    fn test_empty_graph() {
        let graph = UnGraph::<i32, i32>::new_undirected();
        let communities = label_propagation(&graph, 10, Some(0));
        assert!(communities.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = UnGraph::<i32, i32>::new_undirected();
        let a = graph.add_node(0);
        let communities = label_propagation(&graph, 10, Some(0));
        assert_eq!(communities.len(), 1);
        assert!(communities.contains_key(&a));
    }

    #[test]
    fn test_disconnected_graph() {
        let mut graph = UnGraph::<i32, i32>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        // No edges - each node should be its own community
        let communities = label_propagation(&graph, 10, Some(42));
        // Each node keeps its initial label since no neighbors
        assert_ne!(communities[&a], communities[&b]);
        assert_ne!(communities[&b], communities[&c]);
    }

    #[test]
    fn test_star_graph() {
        let mut graph = UnGraph::<i32, i32>::new_undirected();
        let center = graph.add_node(0);
        let leaves: Vec<_> = (0..10).map(|_| graph.add_node(1)).collect();
        for &leaf in &leaves {
            graph.add_edge(center, leaf, 1);
        }

        let communities = label_propagation(&graph, 100, Some(42));

        // Star graph: verify it runs and returns correct number of nodes
        assert_eq!(communities.len(), 11);
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = DiGraph::<i32, i32>::new();
        // Two strongly connected components with one-way bridge
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);

        // Component 1: a <-> b
        graph.add_edge(a, b, 1);
        graph.add_edge(b, a, 1);
        // Component 2: c <-> d
        graph.add_edge(c, d, 1);
        graph.add_edge(d, c, 1);
        // Bridge: b -> c
        graph.add_edge(b, c, 1);

        let communities = label_propagation(&graph, 100, Some(42));
        assert_eq!(communities.len(), 4);
        // b and c are connected via outgoing edges, may end up same community
        // Just verify it runs
        assert!(communities.contains_key(&a));
        assert!(communities.contains_key(&b));
        assert!(communities.contains_key(&c));
        assert!(communities.contains_key(&d));
    }

    #[test]
    fn test_deterministic_with_seed() {
        use hashbrown::HashMap as HashbrownMap;
        use petgraph::visit::NodeIndexable;

        let mut graph = UnGraph::<i32, i32>::new_undirected();
        for i in 0..20 {
            let _ = graph.add_node(i);
        }
        // Add edges for community structure
        for i in 0..19 {
            graph.add_edge(
                petgraph::graph::NodeIndex::new(i),
                petgraph::graph::NodeIndex::new(i + 1),
                1,
            );
        }
        // Additional edges within each half
        for i in 0..8 {
            for j in (i + 2)..10 {
                graph.add_edge(
                    petgraph::graph::NodeIndex::new(i),
                    petgraph::graph::NodeIndex::new(j),
                    1,
                );
            }
        }
        for i in 10..18 {
            for j in (i + 2)..20 {
                graph.add_edge(
                    petgraph::graph::NodeIndex::new(i),
                    petgraph::graph::NodeIndex::new(j),
                    1,
                );
            }
        }

        let communities1 = label_propagation(&graph, 100, Some(123));
        let communities2 = label_propagation(&graph, 100, Some(123));

        // Compare partitions (not raw labels, since label values may differ
        // but the grouping should be identical)
        let make_partition = |communities: &DictMap<_, _>| {
            let mut groups: HashbrownMap<u32, Vec<usize>> = HashbrownMap::new();
            for (&node, &label) in communities {
                groups.entry(label).or_default().push(graph.to_index(node));
            }
            let mut v: Vec<Vec<usize>> = groups
                .into_values()
                .map(|mut vec| {
                    vec.sort();
                    vec
                })
                .collect();
            v.sort();
            v
        };

        assert_eq!(make_partition(&communities1), make_partition(&communities2));
    }

    #[test]
    fn test_convergence() {
        // A complete graph should converge to a single community
        let mut graph = UnGraph::<i32, i32>::new_undirected();
        let nodes: Vec<_> = (0..10).map(|_| graph.add_node(0)).collect();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                graph.add_edge(nodes[i], nodes[j], 1);
            }
        }

        let communities = label_propagation(&graph, 100, Some(42));

        // All nodes should have the same label in a complete graph
        let first_label = communities[&nodes[0]];
        for &node in &nodes[1..] {
            assert_eq!(communities[&node], first_label);
        }
    }
}
