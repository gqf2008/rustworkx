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

//! Girvan-Newman community detection algorithm.
//!
//! The algorithm iteratively removes edges with the highest betweenness
//! centrality, producing a hierarchical decomposition (dendrogram) of
//! the graph into communities.
//!
//! The algorithm is described in:
//! Girvan, M., & Newman, M. E. J. (2002). Community structure in social
//! and biological networks. Proceedings of the National Academy of Sciences,
//! 99(12), 7821-7826.

use hashbrown::HashMap;
use petgraph::visit::{
    EdgeCount, EdgeIndexable, EdgeRef, GraphProp, IntoEdges, IntoNeighborsDirected,
    IntoNodeIdentifiers, NodeCount, NodeIndexable,
};
use petgraph::EdgeType;
use petgraph::stable_graph::StableGraph;

use crate::centrality::edge_betweenness_centrality;
use crate::connectivity::connected_components;
use crate::dictmap::{DictMap, InitWithHasher};

/// Girvan-Newman community detection algorithm.
///
/// This algorithm detects communities by iteratively removing edges with the
/// highest betweenness centrality. At each step, the connected components of
/// the remaining graph define the community partition.
///
/// # Arguments
///
/// * `graph` - The graph to analyze. Edge weights must be f64.
/// * `max_steps` - Maximum number of edges to remove. If None, removes all
///   edges. Each step removes one edge and records the resulting partition.
///
/// # Returns
///
/// A vector of `DictMap`s, where each map represents the community partition
/// at that step. Step 0 is the initial partition (all nodes in community 0),
/// and each subsequent step shows the partition after one more edge removal.
///
/// Community labels are compact integers starting from 0 at each step.
///
/// # Example
///
/// ```rust
/// use petgraph::graph::UnGraph;
/// use rustworkx_core::community::girvan_newman;
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
/// let dendrogram = girvan_newman(&graph, Some(1));
/// // dendrogram[0] = initial partition (all one community)
/// // dendrogram[1] = partition after removing highest-betweenness edge
/// assert_eq!(dendrogram.len(), 2);
/// ```
pub fn girvan_newman<G>(
    graph: G,
    max_steps: Option<usize>,
) -> Vec<DictMap<G::NodeId, u32>>
where
    G: NodeIndexable
        + IntoNodeIdentifiers
        + IntoEdges
        + NodeCount
        + EdgeCount
        + GraphProp
        + EdgeIndexable
        + IntoNeighborsDirected
        + Sync
        + Clone,
    G::NodeId: Eq + std::hash::Hash + Copy + Send,
    G::EdgeId: Eq + Copy + Send,
    G::EdgeWeight: Copy,
    f64: From<G::EdgeWeight>,
    <G as GraphProp>::EdgeType: Sync,
{
    let node_count = graph.node_count();
    if node_count == 0 {
        return vec![];
    }

    // Work on a mutable clone
    let nodes: Vec<G::NodeId> = graph.node_identifiers().collect();
    let n = nodes.len();
    let edge_count = graph.edge_count();
    let max_steps = max_steps.unwrap_or(edge_count).min(edge_count);

    // Collect all edges with their original indices and weights
    #[allow(clippy::type_complexity)]
    let mut working_edges: Vec<(G::EdgeId, G::NodeId, G::NodeId, G::EdgeWeight)> = Vec::new();
    for node in graph.node_identifiers() {
        for edge in graph.edges(node) {
            // For undirected graphs, only collect each edge once
            if !<G as GraphProp>::EdgeType::is_directed() {
                let src = NodeIndexable::to_index(&graph, edge.source());
                let tgt = NodeIndexable::to_index(&graph, edge.target());
                if src > tgt {
                    continue;
                }
            }
            working_edges.push((edge.id(), edge.source(), edge.target(), *edge.weight()));
        }
    }

    // Build a StableGraph with f64 weights for the algorithm
    let mut sg = StableGraph::<(), f64, <G as GraphProp>::EdgeType>::with_capacity(n, edge_count);
    let mut node_map: HashMap<G::NodeId, petgraph::graph::NodeIndex> = HashMap::new();
    for &node in &nodes {
        let idx = sg.add_node(());
        node_map.insert(node, idx);
    }
    for &(_eid, src, tgt, weight) in &working_edges {
        let src_idx = node_map[&src];
        let tgt_idx = node_map[&tgt];
        let w: f64 = f64::from(weight);
        sg.add_edge(src_idx, tgt_idx, w);
    }

    // Compute initial partition (all in one community)
    let mut dendrogram: Vec<DictMap<G::NodeId, u32>> = Vec::with_capacity(max_steps + 1);
    let mut initial = DictMap::with_capacity(n);
    for &node in &nodes {
        initial.insert(node, 0);
    }
    dendrogram.push(initial);

    for _step in 0..max_steps {
        // Compute edge betweenness
        let betweenness = edge_betweenness_centrality(&sg, false, 256);

        // Find edge with maximum betweenness among remaining edges
        let mut max_betweenness = -1.0f64;
        let mut max_edge_idx: Option<usize> = None;
        for (i, &b) in betweenness.iter().enumerate() {
            if let Some(b_val) = b {
                if b_val > max_betweenness {
                    max_betweenness = b_val;
                    max_edge_idx = Some(i);
                }
            }
        }

        let edge_to_remove = match max_edge_idx {
            Some(idx) => idx,
            None => break, // No more edges to remove
        };

        // Remove the edge from the StableGraph
        sg.remove_edge(petgraph::graph::EdgeIndex::new(edge_to_remove));

        // Compute connected components for the new partition
        let components = connected_components(&sg);
        let mut partition = DictMap::with_capacity(n);
        for (comm_id, component) in components.iter().enumerate() {
            for &sg_node in component {
                // Map back to original node ID
                let original_node = nodes[sg_node.index()];
                partition.insert(original_node, comm_id as u32);
            }
        }
        dendrogram.push(partition);
    }

    dendrogram
}

#[cfg(test)]
mod tests {
    use petgraph::graph::UnGraph;

    use super::girvan_newman;

    #[test]
    fn test_empty_graph() {
        let graph = UnGraph::<i32, f64>::new_undirected();
        let dendrogram = girvan_newman(&graph, None);
        assert!(dendrogram.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let dendrogram = girvan_newman(&graph, None);
        assert_eq!(dendrogram.len(), 1);
        assert_eq!(dendrogram[0][&a], 0);
    }

    #[test]
    fn test_two_communities() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        let d = graph.add_node(3);
        let e = graph.add_node(4);
        let f = graph.add_node(5);

        // Community 1: triangle
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(a, c, 1.0);
        // Community 2: triangle
        graph.add_edge(d, e, 1.0);
        graph.add_edge(e, f, 1.0);
        graph.add_edge(d, f, 1.0);
        // Bridge
        graph.add_edge(c, d, 1.0);

        let dendrogram = girvan_newman(&graph, Some(1));
        assert_eq!(dendrogram.len(), 2);

        // Initial: all one community
        assert_eq!(dendrogram[0].len(), 6);

        // After removing bridge: should be 2 communities
        let partition = &dendrogram[1];
        // a, b, c should be in same community
        assert_eq!(partition[&a], partition[&b]);
        assert_eq!(partition[&b], partition[&c]);
        // d, e, f should be in same community
        assert_eq!(partition[&d], partition[&e]);
        assert_eq!(partition[&e], partition[&f]);
        // Bridge communities should be separate
        assert_ne!(partition[&a], partition[&d]);
    }

    #[test]
    fn test_max_steps() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        let c = graph.add_node(2);
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);

        // With max_steps=2 (2 edges), should get 3 partitions
        let dendrogram = girvan_newman(&graph, Some(2));
        assert_eq!(dendrogram.len(), 3);

        // Last partition should have each node in its own community
        let last = &dendrogram[2];
        assert_ne!(last[&a], last[&b]);
        assert_ne!(last[&b], last[&c]);
    }

    #[test]
    fn test_complete_graph() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let nodes: Vec<_> = (0..6).map(|_| graph.add_node(0)).collect();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                graph.add_edge(nodes[i], nodes[j], 1.0);
            }
        }

        // Complete graph: removing one edge shouldn't split it
        let dendrogram = girvan_newman(&graph, Some(1));
        assert_eq!(dendrogram.len(), 2);
        // After removing one edge, still connected
        let labels: Vec<u32> = dendrogram[1].values().copied().collect();
        assert_eq!(labels.iter().min(), Some(&0));
        assert_eq!(labels.iter().max(), Some(&0));
    }

    #[test]
    fn test_dendrogram_length() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let b = graph.add_node(1);
        graph.add_edge(a, b, 1.0);

        // 1 edge: initial + 1 removal = 2 partitions
        let dendrogram = girvan_newman(&graph, None);
        assert_eq!(dendrogram.len(), 2);

        // After removal: each node isolated
        assert_ne!(dendrogram[1][&a], dendrogram[1][&b]);
    }
}
