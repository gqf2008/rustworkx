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

use hashbrown::{HashMap, HashSet};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};

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
/// let communities = leiden_communities(&graph, None, None, None);
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
    seed: Option<u64>,
) -> DictMap<G::NodeId, u32>
where
    G: NodeIndexable + IntoNodeIdentifiers + IntoEdges + NodeCount,
    G::NodeId: std::cmp::Eq + std::hash::Hash + Copy + Send + Sync,
    G::EdgeWeight: Copy,
    f64: From<G::EdgeWeight>,
{
    let max_iterations = max_iterations.unwrap_or(100);
    let resolution = resolution.unwrap_or(1.0);
    let mut seed = seed.unwrap_or(42);

    let node_count = graph.node_count();
    if node_count == 0 {
        return DictMap::new();
    }

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

    let m = total_weight / 2.0;

    if m == 0.0 {
        let mut result = DictMap::with_capacity(n);
        for (i, &node) in nodes.iter().enumerate() {
            result.insert(node, i as u32);
        }
        return result;
    }

    // Each node starts in its own community
    let mut node_to_community: Vec<u32> = (0..n).map(|i| i as u32).collect();

    // Compute node degrees (sum of incident edge weights, excluding self-loops)
    let mut node_degree: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for &(j, w) in &adj[i] {
            if j != i {
                node_degree[i] += w;
            }
        }
    }

    // Run Leiden iterations
    leiden_pass(
        &mut node_to_community,
        &adj,
        &node_degree,
        m,
        resolution,
        max_iterations,
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

fn rand_u64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

fn leiden_pass(
    node_to_community: &mut [u32],
    adj: &[Vec<(usize, f64)>],
    node_degree: &[f64],
    m: f64,
    resolution: f64,
    max_iterations: usize,
    seed: &mut u64,
) {
    let n = adj.len();
    let two_m = 2.0 * m;

    for _iteration in 0..max_iterations {
        // Phase 1: Local moving
        local_move(
            node_to_community,
            adj,
            node_degree,
            m,
            two_m,
            resolution,
            seed,
        );

        // Phase 2: Refinement — split communities into well-connected sub-communities
        let mut refined: Vec<u32> = (0..(n as u32)).collect();
        refine_communities(
            &mut refined,
            node_to_community,
            adj,
            m,
            two_m,
            resolution,
            seed,
        );

        // Phase 3: Aggregate using refined communities
        let changed = aggregate(node_to_community, &refined, n);

        if !changed {
            break;
        }
    }
}

/// Local moving phase — same as Louvain
fn local_move(
    node_to_community: &mut [u32],
    adj: &[Vec<(usize, f64)>],
    node_degree: &[f64],
    m: f64,
    two_m: f64,
    resolution: f64,
    seed: &mut u64,
) {
    let n = adj.len();

    // Compute community statistics
    let mut k_tot: HashMap<u32, f64> = HashMap::new();
    let mut k_in: HashMap<u32, f64> = HashMap::new();

    for i in 0..n {
        let c = node_to_community[i];
        *k_tot.entry(c).or_insert(0.0) += node_degree[i];
    }
    for i in 0..n {
        let ci = node_to_community[i];
        for &(j, w) in &adj[i] {
            if j != i && node_to_community[j] == ci {
                *k_in.entry(ci).or_insert(0.0) += w;
            }
        }
    }

    for _pass in 0..n {
        let mut improved = false;

        let mut order: Vec<usize> = (0..n).collect();
        fisher_yates_shuffle(&mut order, seed);

        for &i in &order {
            let ci = node_to_community[i];
            let ki = node_degree[i];
            if ki == 0.0 {
                continue;
            }

            // Remove node from its community
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
                let dq = ki_to_comm / m - resolution * ktot * ki / (two_m * m);
                if dq > best_dq {
                    best_dq = dq;
                    best_comm = comm;
                }
            }

            // Place node back
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

/// Refinement phase — split each community into well-connected sub-communities.
///
/// Following the Leiden paper (Traag et al. 2019), this works by:
/// 1. Starting with each node in its own sub-community within its community.
/// 2. For each community, find connected components via BFS.
///    Each connected component becomes a distinct refined sub-community.
///    This guarantees all refined communities are connected.
fn refine_communities(
    refined: &mut [u32],
    node_to_community: &[u32],
    adj: &[Vec<(usize, f64)>],
    m: f64,
    two_m: f64,
    resolution: f64,
    seed: &mut u64,
) {
    let n = adj.len();

    // Group nodes by community
    let mut community_nodes: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &c) in node_to_community.iter().enumerate() {
        community_nodes.entry(c).or_default().push(i);
    }

    let mut next_refined_id = n as u32;

    // Reusable buffer for sub-community assignments — avoids O(n) allocation per component
    let mut sub_comm: Vec<u32> = vec![0; n];

    for (_comm, nodes) in &community_nodes {
        if nodes.is_empty() {
            continue;
        }
        if nodes.len() == 1 {
            refined[nodes[0]] = next_refined_id;
            next_refined_id += 1;
            continue;
        }

        // Build a HashSet for O(1) membership testing within this community
        let node_set: HashSet<usize> = nodes.iter().copied().collect();

        // Find connected components within this community via BFS
        let mut visited: HashSet<usize> = HashSet::new();
        let mut components: Vec<Vec<usize>> = Vec::new();

        for &start in nodes {
            if visited.contains(&start) {
                continue;
            }
            let mut component: Vec<usize> = Vec::new();
            let mut queue: Vec<usize> = vec![start];
            visited.insert(start);

            while let Some(node) = queue.pop() {
                component.push(node);
                for &(nbr, _w) in &adj[node] {
                    if node_set.contains(&nbr) && !visited.contains(&nbr) {
                        visited.insert(nbr);
                        queue.push(nbr);
                    }
                }
            }

            components.push(component);
        }

        // For each connected component, run local moving to find sub-communities
        for component in components {
            if component.len() == 1 {
                refined[component[0]] = next_refined_id;
                next_refined_id += 1;
                continue;
            }

            // Build HashSet for the component
            let comp_set: HashSet<usize> = component.iter().copied().collect();

            // Start: each node in its own sub-community
            // Reuse sub_comm buffer — only reset entries for nodes in this component afterward
            for (idx, &node) in component.iter().enumerate() {
                sub_comm[node] = idx as u32;
            }

            let mut sub_k_tot: HashMap<u32, f64> = HashMap::new();
            let mut sub_k_in: HashMap<u32, f64> = HashMap::new();

            for &node in &component {
                let sc = sub_comm[node];

                // Compute degree restricted to this component
                let ki: f64 = adj[node]
                    .iter()
                    .filter(|&&(j, _)| comp_set.contains(&j) && j != node)
                    .map(|&(_, w)| w)
                    .sum();
                *sub_k_tot.entry(sc).or_insert(0.0) += ki;

                // Compute intra-sub-community edge weight
                let ki_in: f64 = adj[node]
                    .iter()
                    .filter(|&&(j, _)| j != node && comp_set.contains(&j) && sub_comm[j] == sc)
                    .map(|&(_, w)| w)
                    .sum();
                *sub_k_in.entry(sc).or_insert(0.0) += ki_in;
            }

            // Local moving within this connected component
            let mut order = component.clone();
            for _pass in 0..component.len() {
                fisher_yates_shuffle(&mut order, seed);
                let mut improved = false;

                for &i in &order {
                    let sci = sub_comm[i];

                    // Compute degree restricted to this component
                    let ki: f64 = adj[i]
                        .iter()
                        .filter(|&&(j, _)| comp_set.contains(&j) && j != i)
                        .map(|&(_, w)| w)
                        .sum();
                    if ki == 0.0 {
                        continue;
                    }

                    // Remove from sub-community
                    let ki_in =
                        edge_weight_to_community_with_set(i, sci, adj, &sub_comm, &comp_set);
                    let ki_self = self_loop_weight(i, adj);
                    *sub_k_in.entry(sci).or_insert(0.0) -= ki_in + ki_self;
                    *sub_k_tot.entry(sci).or_insert(0.0) -= ki;

                    // Find neighboring sub-communities
                    let mut neighbor_sub: HashMap<u32, f64> = HashMap::new();
                    for &(j, w) in &adj[i] {
                        if j == i || !comp_set.contains(&j) {
                            continue;
                        }
                        *neighbor_sub.entry(sub_comm[j]).or_insert(0.0) += w;
                    }

                    let mut best_sc = sci;
                    let mut best_dq = 0.0;

                    for (&sc, &w) in &neighbor_sub {
                        let ktot = *sub_k_tot.get(&sc).unwrap_or(&0.0);
                        let dq = w / m - resolution * ktot * ki / (two_m * m);
                        if dq > best_dq {
                            best_dq = dq;
                            best_sc = sc;
                        }
                    }

                    sub_comm[i] = best_sc;
                    let new_ki_in =
                        edge_weight_to_community_with_set(i, best_sc, adj, &sub_comm, &comp_set);
                    let new_ki_self = self_loop_weight(i, adj);
                    *sub_k_in.entry(best_sc).or_insert(0.0) += new_ki_in + new_ki_self;
                    *sub_k_tot.entry(best_sc).or_insert(0.0) += ki;

                    if best_sc != sci {
                        improved = true;
                    }
                }

                if !improved {
                    break;
                }
            }

            // Map sub-communities to refined IDs
            let mut sub_to_refined: HashMap<u32, u32> = HashMap::new();
            for &node in &component {
                let sc = sub_comm[node];
                let rid = sub_to_refined.entry(sc).or_insert_with(|| {
                    let id = next_refined_id;
                    next_refined_id += 1;
                    id
                });
                refined[node] = *rid;
            }

            // Reset sub_comm entries for reuse by next component
            for &node in &component {
                sub_comm[node] = 0;
            }
        }
    }
}

/// Aggregation phase — assign each node to the community of its refined group.
///
/// Returns `true` if any node changed community assignment.
fn aggregate(node_to_community: &mut [u32], refined: &[u32], n: usize) -> bool {
    // Map each distinct refined ID to a new compact community ID
    let mut refined_to_new: HashMap<u32, u32> = HashMap::new();
    let mut next_comm: u32 = 0;

    let mut changed = false;
    for i in 0..n {
        let r = refined[i];
        let new_comm = refined_to_new.entry(r).or_insert_with(|| {
            let id = next_comm;
            next_comm += 1;
            id
        });
        if *new_comm != node_to_community[i] {
            changed = true;
            node_to_community[i] = *new_comm;
        }
    }

    changed
}

/// Sum of edge weights from node to nodes in community (excluding node itself)
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

/// Sum of edge weights from node to nodes in a sub-community,
/// using a Vec lookup and a component membership set.
fn edge_weight_to_community_with_set(
    node: usize,
    comm: u32,
    adj: &[Vec<(usize, f64)>],
    sub_comm: &[u32],
    comp_set: &HashSet<usize>,
) -> f64 {
    let mut sum = 0.0;
    for &(j, w) in &adj[node] {
        if j != node && comp_set.contains(&j) && sub_comm[j] == comm {
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

fn fisher_yates_shuffle(slice: &mut [usize], seed: &mut u64) {
    let n = slice.len();
    for i in (1..n).rev() {
        let r = (rand_u64(seed) as usize) % (i + 1);
        slice.swap(i, r);
    }
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

        let communities = leiden_communities(&graph, None, None, Some(42));

        assert_eq!(communities[&a], communities[&b]);
        assert_eq!(communities[&b], communities[&c]);
        assert_eq!(communities[&d], communities[&e]);
        assert_eq!(communities[&e], communities[&f]);
        assert_ne!(communities[&a], communities[&d]);
    }

    #[test]
    fn test_empty_graph() {
        let graph = UnGraph::<i32, f64>::new_undirected();
        let communities = leiden_communities(&graph, None, None, None);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        let a = graph.add_node(0);
        let communities = leiden_communities(&graph, None, None, None);
        assert_eq!(communities.len(), 1);
        assert!(communities.contains_key(&a));
    }

    #[test]
    fn test_no_edges() {
        let mut graph = UnGraph::<i32, f64>::new_undirected();
        for _ in 0..5 {
            graph.add_node(0);
        }
        let communities = leiden_communities(&graph, None, None, None);
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

        let communities = leiden_communities(&graph, None, None, Some(42));
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

        let communities = leiden_communities(&graph, None, None, Some(42));

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

        let communities = leiden_communities(&graph, None, None, None);
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

        let communities = leiden_communities(&graph, None, None, Some(42));
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

        let communities = leiden_communities(&graph, None, None, Some(42));
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

        let communities = leiden_communities(&graph, None, None, Some(42));
        // Should not panic; all nodes should be assigned
        assert_eq!(communities.len(), 4);
    }
}
