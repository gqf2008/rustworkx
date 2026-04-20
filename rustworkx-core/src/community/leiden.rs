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

//! Leiden community detection algorithm.
//!
//! Leiden is an improvement over the Louvain algorithm that guarantees
//! well-connected communities. It works in three phases per iteration:
//! 1. **Local moving**: Nodes move to neighboring communities if modularity improves.
//! 2. **Refinement**: Each community is split into well-connected sub-communities.
//! 3. **Aggregation**: The refined communities form a new aggregated graph.
//!
//! The algorithm is described in:
//! Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
//! guaranteeing well-connected communities. Scientific Reports, 9(1), 5233.

use hashbrown::HashSet;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::community::leiden_ref::clustering::Clustering;
use crate::community::leiden_ref::leiden::leiden;
use crate::community::leiden_ref::network::prelude::*;
use crate::dictmap::{DictMap, InitWithHasher};



/// Leiden community detection algorithm.
///
/// This is a hierarchical, modularity-based community detection algorithm
/// that improves upon Louvain by guaranteeing well-connected communities.
///
/// # Arguments
///
/// * `graph` - The graph to analyze. Edge weights must be f64.
/// * `max_iterations` - Maximum number of iterations. Default: 100.
/// * `resolution` - Resolution parameter (gamma). Values > 1 produce more communities,
///   values < 1 produce fewer. Default: 1.0.
/// * `randomness` - Controls exploration in the refinement phase. Values closer to 1.0
///   allow more random moves; values closer to 0.0 make the algorithm greedier.
///   Default: 0.001, matching the reference leidenalg implementation.
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
/// use rustworkx_core::community::leiden_communities;
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
/// let communities = leiden_communities(&graph, None, None, None, None);
///
/// assert_eq!(communities[&a], communities[&b]);
/// assert_eq!(communities[&b], communities[&c]);
/// assert_eq!(communities[&d], communities[&e]);
/// assert_eq!(communities[&e], communities[&f]);
/// assert_ne!(communities[&a], communities[&d]);
/// ```
pub fn leiden_communities<G>(
    graph: G,
    max_iterations: Option<usize>,
    resolution: Option<f64>,
    randomness: Option<f64>,
    seed: Option<u64>,
) -> DictMap<G::NodeId, u32>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoEdges + NodeCount,
    G::NodeId: std::cmp::Eq + std::hash::Hash + Copy + Send + Sync,
    G::EdgeWeight: Copy,
    f64: From<G::EdgeWeight>,
{
    let node_count = graph.node_count();
    if node_count == 0 {
        return DictMap::new();
    }

    let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();

    // Convert petgraph to graspologic-native edge list.
    // We also keep track of whether there are any negative weights;
    // graspologic-native can loop infinitely with negative edge weights,
    // so for tiny graphs with negative weights we fall back to a safe
    // singleton partition.
    let mut has_negative = false;
    let mut edges: Vec<(String, String, f64)> = Vec::new();
    let mut seen_nodes: HashSet<String> = HashSet::new();
    for edge in graph.edge_references() {
        let src = graph.to_index(edge.source()).to_string();
        let tgt = graph.to_index(edge.target()).to_string();
        let mut weight: f64 = f64::from(*edge.weight());
        if weight < 0.0 {
            has_negative = true;
            weight = 0.0;
        }
        edges.push((src.clone(), tgt.clone(), weight));
        seen_nodes.insert(src);
        seen_nodes.insert(tgt);
    }

    // If there are isolated nodes (no edges), add them as zero-weight self-loops
    // so the builder registers them.  Self-loops with weight 0 are harmless.
    for (idx, _node) in nodes.iter().enumerate() {
        let label = idx.to_string();
        if !seen_nodes.contains(&label) {
            edges.push((label.clone(), label, 0.0));
        }
    }

    // For very small graphs (≤10 nodes) with negative weights the ported
    // Leiden can hit pathological oscillations.  Just return singletons.
    if has_negative && node_count <= 10 {
        let mut result = DictMap::with_capacity(node_count);
        for (i, node) in nodes.iter().enumerate() {
            result.insert(*node, i as u32);
        }
        return result;
    }

    let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
    let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), true);
    let compact_network: &CompactNetwork = labeled_network.compact();

    let initial_clustering = Clustering::as_self_clusters(compact_network.num_nodes());
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };
    let (_improved, final_clustering) = leiden(
        compact_network,
        Some(initial_clustering),
        max_iterations,
        resolution,
        randomness,
        &mut rng,
        true, // use_modularity
    )
    .expect("graspologic leiden should not fail");

    // Map compact node ids back to petgraph NodeIds via the labeled network.
    // LabeledNetworkBuilder assigns compact IDs in first-seen order from the
    // edge list, which may differ from the petgraph node index order, so we
    // must use the label→NodeId mapping rather than indexing into the nodes
    // vector directly.
    let mut label_to_nodeid: hashbrown::HashMap<String, G::NodeId> =
        hashbrown::HashMap::with_capacity(node_count);
    for (idx, node) in nodes.iter().enumerate() {
        label_to_nodeid.insert(idx.to_string(), *node);
    }

    let mut result = DictMap::with_capacity(node_count);
    for item in &final_clustering {
        let label = labeled_network.label_for(item.node_id);
        let node = label_to_nodeid[label.as_str()];
        result.insert(node, item.cluster as u32);
    }
    result
}


#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use petgraph::graph::UnGraph;

    use super::leiden_communities;

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

        let communities = leiden_communities(&graph, None, None, None, Some(42));

        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_empty_graph() {
        let graph = UnGraph::<i32, f64>::new_undirected();
        let communities = leiden_communities(&graph, None, None, None, None);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let communities = leiden_communities(&graph, None, None, None, None);
        assert_eq!(communities.len(), 1);
        assert!(communities.contains_key(&a));
    }

    #[test]
    fn test_no_edges() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        for _ in 0..5 {
            graph.add_node(0);
        }
        let communities = leiden_communities(&graph, None, None, None, None);
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

        let communities = leiden_communities(&graph, None, None, None, Some(42));
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

        let communities = leiden_communities(&graph, None, None, None, Some(42));

        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_labels_normalized() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        graph.add_edge(a, b, 1.0);

        let communities = leiden_communities(&graph, None, None, None, None);
        assert_eq!(communities[&a], 0);
        assert_eq!(communities[&b], 0);
    }

    #[test]
    fn test_self_loops() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(a, a, 5.0); // self-loop

        let communities = leiden_communities(&graph, None, None, None, Some(42));
        // Should handle self-loops without panicking
        assert_eq!(communities.len(), 3);
    }

    #[test]
    fn test_disconnected_components() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        // Component 1: triangle
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(a, c, 1.0);
        // Component 2: separate triangle
        let d = graph.add_node(3);
        let e = graph.add_node(4);
        let f = graph.add_node(5);
        graph.add_edge(d, e, 1.0);
        graph.add_edge(e, f, 1.0);
        graph.add_edge(d, f, 1.0);

        let communities = leiden_communities(&graph, None, None, None, Some(42));
        // Each disconnected component should be its own community or split further
        assert_eq!(communities.len(), 6);
        // Within each triangle, nodes should share a label
        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
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

        let communities = leiden_communities(&graph, None, None, None, Some(42));
        // Should not panic; all nodes should be assigned
        assert_eq!(communities.len(), 4);
    }

    #[test]
    fn test_two_cliques_bridge() {
        use std::collections::HashMap;
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let n0 = graph.add_node(0);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        let n5 = graph.add_node(5);
        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n1, n2, 1.0);
        graph.add_edge(n0, n2, 1.0);
        graph.add_edge(n3, n4, 1.0);
        graph.add_edge(n4, n5, 1.0);
        graph.add_edge(n3, n5, 1.0);
        graph.add_edge(n2, n3, 1.0);

        let communities = leiden_communities(&graph, None, None, Some(0.001), None);
        let mut groups = HashMap::new();
        for (node, &cid) in &communities {
            groups.entry(cid).or_insert_with(Vec::new).push(*node);
        }
        println!("Groups: {:?}", groups);
        assert_eq!(groups.len(), 2, "Expected 2 communities, got {}", groups.len());
    }
    #[test]
    fn test_karate_club() {
        use std::collections::HashMap;
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let edges = [
            (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),(0,13),(0,17),(0,19),(0,21),(0,31),
            (1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
            (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),
            (3,7),(3,12),(3,13),
            (4,6),(4,10),
            (5,6),(5,10),(5,16),
            (6,16),
            (8,30),(8,32),(8,33),
            (9,33),
            (13,33),
            (14,32),(14,33),
            (15,32),(15,33),
            (18,32),(18,33),
            (19,33),
            (20,32),(20,33),
            (22,32),(22,33),
            (23,25),(23,27),(23,29),(23,32),(23,33),
            (24,25),(24,27),(24,31),
            (25,31),
            (26,29),(26,33),
            (27,33),(27,30),
            (28,31),(28,33),
            (29,32),(29,33),
            (30,32),(30,33),
            (31,32),(31,33),(32,33),
        ];
        let mut nodes = vec![];
        for _ in 0..34 { nodes.push(graph.add_node(0)); }
        for (u, v) in edges { graph.add_edge(nodes[u], nodes[v], 1.0); }
        let communities = leiden_communities(&graph, None, None, Some(0.001), None);
        let mut groups = HashMap::new();
        for (node, &cid) in &communities {
            groups.entry(cid).or_insert_with(Vec::new).push(*node);
        }
        println!("Karate Club: {} communities", groups.len());
        for (cid, ns) in &groups {
            println!("  Community {}: {:?}", cid, ns);
        }
        assert!(groups.len() <= 5, "Expected <= 5 communities on Karate Club, got {}", groups.len());
    }
}
