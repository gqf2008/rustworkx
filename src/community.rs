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

use pyo3::prelude::*;
use pyo3::types::PyDict;

use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::{EdgeRef, NodeIndexable};

use rustworkx_core::community::label_propagation as core_label_propagation;
use rustworkx_core::community::louvain_communities as core_louvain;

use crate::{digraph, graph};

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
/// :param graph: The input graph (PyGraph or PyDiGraph) to analyze
/// :param max_iterations: Maximum number of iterations to perform. Default: 100.
///     The algorithm will also stop early if no labels change in an iteration.
/// :param seed: Optional random seed for reproducibility. When not provided,
///     results may vary between runs due to random tie-breaking and node ordering.
///
/// :returns: A dictionary mapping node indices to community labels (integers).
///     Nodes in the same community share the same label.
/// :rtype: dict
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///     from rustworkx.visualization import mpl_draw
///
///     # Create a graph with two communities
///     graph = rx.PyGraph()
///     # Community 1
///     a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
///     graph.add_edge(a, b, 1)
///     graph.add_edge(b, c, 1)
///     graph.add_edge(a, c, 1)
///     # Community 2
///     d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
///     graph.add_edge(d, e, 1)
///     graph.add_edge(e, f, 1)
///     graph.add_edge(d, f, 1)
///     # Bridge between communities
///     graph.add_edge(c, d, 1)
///
///     communities = rx.label_propagation(graph, seed=42)
///
///     # Group nodes by community
///     community_groups = {}
///     for node, label in communities.items():
///         community_groups.setdefault(label, []).append(node)
///     print(f"Number of communities: {len(community_groups)}")
///
///     # Color nodes by community for visualization
///     colors = [communities[node] for node in graph.node_indices()]
///     mpl_draw(graph, node_color=colors, pos=rx.spring_layout(graph, seed=42))
///
#[pyfunction]
#[pyo3(signature = (graph, /, max_iterations=100, seed=None))]
pub fn graph_label_propagation(
    py: Python,
    graph: &graph::PyGraph,
    max_iterations: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let graph_clone = graph.graph.clone();
    let result = core_label_propagation(&graph_clone, max_iterations, seed);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(node.index(), label)?;
    }
    Ok(out_dict.into())
}

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
/// :param graph: The input graph (PyDiGraph) to analyze
/// :param max_iterations: Maximum number of iterations to perform. Default: 100.
///     The algorithm will also stop early if no labels change in an iteration.
/// :param seed: Optional random seed for reproducibility. When not provided,
///     results may vary between runs due to random tie-breaking and node ordering.
///
/// :returns: A dictionary mapping node indices to community labels (integers).
///     Nodes in the same community share the same label.
/// :rtype: dict
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///
///     # Create a directed graph with two communities
///     graph = rx.PyDiGraph()
///     # Community 1
///     a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
///     graph.add_edge(a, b, 1)
///     graph.add_edge(b, c, 1)
///     graph.add_edge(c, a, 1)
///     # Community 2
///     d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
///     graph.add_edge(d, e, 1)
///     graph.add_edge(e, f, 1)
///     graph.add_edge(f, d, 1)
///     # Bridge between communities
///     graph.add_edge(c, d, 1)
///
///     communities = rx.digraph_label_propagation(graph, seed=42)
///     print(communities)
///
#[pyfunction]
#[pyo3(signature = (graph, /, max_iterations=100, seed=None))]
pub fn digraph_label_propagation(
    py: Python,
    graph: &digraph::PyDiGraph,
    max_iterations: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let graph_clone = graph.graph.clone();
    let result = core_label_propagation(&graph_clone, max_iterations, seed);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(node.index(), label)?;
    }
    Ok(out_dict.into())
}

/// Louvain community detection algorithm.
///
/// This is a hierarchical, modularity-based community detection algorithm.
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
/// :param graph: The input graph (PyGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param max_levels: Maximum number of hierarchical levels to process.
///     Default: 100.
/// :param resolution: Resolution parameter (gamma). Values > 1 produce more
///     communities, values < 1 produce fewer. Default: 1.0.
///
/// :returns: A dictionary mapping node indices to community labels (integers).
///     Nodes in the same community share the same label.
/// :rtype: dict
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///
///     # Create a graph with two clear communities
///     graph = rx.PyGraph()
///     # Community 1: fully connected triangle
///     a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
///     graph.add_edge(a, b, 1.0)
///     graph.add_edge(b, c, 1.0)
///     graph.add_edge(a, c, 1.0)
///     # Community 2: fully connected triangle
///     d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
///     graph.add_edge(d, e, 1.0)
///     graph.add_edge(e, f, 1.0)
///     graph.add_edge(d, f, 1.0)
///     # Bridge between communities
///     graph.add_edge(c, d, 1.0)
///
///     communities = rx.louvain_communities(graph)
///     print(communities)
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, max_levels=None, resolution=None))]
pub fn graph_louvain_communities(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    max_levels: Option<usize>,
    resolution: Option<f64>,
) -> PyResult<Py<PyAny>> {
    // Build a new graph with f64 weights
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_louvain(&weighted_graph, max_levels, resolution);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Louvain community detection algorithm for directed graphs.
///
/// This is a hierarchical, modularity-based community detection algorithm.
/// It works in two phases:
/// 1. **Local moving**: Each node is moved to the community of a neighbor if it
///    improves the modularity. This is repeated until no improvement is possible.
/// 2. **Aggregation**: Nodes in the same community are contracted into supernodes,
///    and the process repeats on the aggregated graph.
///
/// The algorithm is described in:
/// Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008).
/// Fast unfolding of communities in large networks. Journal of Statistical
/// Mechanics: Theory and Experiment, P10008.
///
/// :param graph: The input graph (PyDiGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param max_levels: Maximum number of hierarchical levels to process.
///     Default: 100.
/// :param resolution: Resolution parameter (gamma). Values > 1 produce more
///     communities, values < 1 produce fewer. Default: 1.0.
///
/// :returns: A dictionary mapping node indices to community labels (integers).
///     Nodes in the same community share the same label.
/// :rtype: dict
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///
///     # Create a directed graph with two clear communities
///     graph = rx.PyDiGraph()
///     # Community 1
///     a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
///     graph.add_edge(a, b, 1.0)
///     graph.add_edge(b, c, 1.0)
///     graph.add_edge(c, a, 1.0)
///     # Community 2
///     d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
///     graph.add_edge(d, e, 1.0)
///     graph.add_edge(e, f, 1.0)
///     graph.add_edge(f, d, 1.0)
///     # Bridge between communities
///     graph.add_edge(c, d, 1.0)
///
///     communities = rx.digraph_louvain_communities(graph)
///     print(communities)
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, max_levels=None, resolution=None))]
pub fn digraph_louvain_communities(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    max_levels: Option<usize>,
    resolution: Option<f64>,
) -> PyResult<Py<PyAny>> {
    // Build a new graph with f64 weights
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_louvain(&weighted_graph, max_levels, resolution);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Helper: convert a StablePyGraph (Py<PyAny> edge weights) to a StableGraph with f64 weights
fn build_f64_graph<Ty>(
    py: Python,
    src: &StableGraph<Py<PyAny>, Py<PyAny>, Ty>,
    weight_fn: &Option<Py<PyAny>>,
    default_weight: f64,
) -> PyResult<StableGraph<(), f64, Ty>>
where
    Ty: petgraph::EdgeType,
{
    use crate::weight_callable;

    let mut out = StableGraph::<(), f64, Ty>::with_capacity(src.node_count(), src.edge_count());

    // Map old node indices to new node indices (preserving order, skipping removed)
    let mut old_to_new: Vec<Option<NodeIndex>> = vec![None; src.node_bound()];
    for old_node in src.node_indices() {
        let new_node = out.add_node(());
        old_to_new[old_node.index()] = Some(new_node);
    }

    // Use edge_iter via IntoEdges trait
    for node_idx in src.node_indices() {
        for edge in src.edges(node_idx) {
            let src_new = old_to_new[node_idx.index()].unwrap();
            let tgt_new = old_to_new[edge.target().index()].unwrap();
            let w: f64 = weight_callable(py, weight_fn, edge.weight(), default_weight)?;
            // Only add each edge once (source < target for undirected)
            if Ty::is_directed() || src_new.index() <= tgt_new.index() {
                out.add_edge(src_new, tgt_new, w);
            }
        }
    }

    Ok(out)
}
