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

use rustworkx_core::community::girvan_newman as core_girvan_newman;
use rustworkx_core::community::greedy_modularity_communities as core_greedy_modularity;
use rustworkx_core::community::infomap_communities as core_infomap;
use rustworkx_core::community::label_propagation as core_label_propagation;
use rustworkx_core::community::leiden_communities as core_leiden;
use rustworkx_core::community::louvain_communities as core_louvain;
use rustworkx_core::community::walktrap_communities as core_walktrap;
use rustworkx_core::dictmap::{DictMap, InitWithHasher};

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
/// :param max_pass_iterations: Maximum number of inner local-moving iterations
///     per level. Default: 100.
/// :param resolution: Resolution parameter (gamma). Values > 1 produce more
///     communities, values < 1 produce fewer. Default: 1.0.
/// :param seed: Optional random seed for reproducibility. When not provided,
///     results may vary between runs due to random node ordering.
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
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, max_levels=None, max_pass_iterations=None, resolution=None, seed=None))]
pub fn graph_louvain_communities(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    max_levels: Option<usize>,
    max_pass_iterations: Option<usize>,
    resolution: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    // Build a new graph with f64 weights
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_louvain(&weighted_graph, max_levels, max_pass_iterations, resolution, seed);

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
/// :param max_pass_iterations: Maximum number of inner local-moving iterations
///     per level. Default: 100.
/// :param resolution: Resolution parameter (gamma). Values > 1 produce more
///     communities, values < 1 produce fewer. Default: 1.0.
/// :param seed: Optional random seed for reproducibility. When not provided,
///     results may vary between runs due to random node ordering.
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
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, max_levels=None, max_pass_iterations=None, resolution=None, seed=None))]
pub fn digraph_louvain_communities(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    max_levels: Option<usize>,
    max_pass_iterations: Option<usize>,
    resolution: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    // Build a new graph with f64 weights
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_louvain(&weighted_graph, max_levels, max_pass_iterations, resolution, seed);

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

/// Leiden community detection algorithm.
///
/// This is a hierarchical, modularity-based community detection algorithm
/// that improves upon Louvain by guaranteeing well-connected communities.
/// It works in three phases per iteration:
/// 1. **Local moving**: Each node is moved to the community of a neighbor if it
///    improves the modularity. This is repeated until no improvement is possible.
/// 2. **Refinement**: Each community is split into well-connected sub-communities,
///    ensuring the final communities are connected.
/// 3. **Aggregation**: The refined communities form a new aggregated graph, and
///    the process repeats.
///
/// The algorithm terminates when no further modularity improvement is possible.
///
/// The algorithm is described in:
/// Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
/// guaranteeing well-connected communities. Scientific Reports, 9(1), 5233.
///
/// :param graph: The input graph (PyGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param max_iterations: Maximum number of iterations to perform.
///     Default: 100.
/// :param resolution: Resolution parameter (gamma). Values > 1 produce more
///     communities, values < 1 produce fewer. Default: 1.0.
/// :param seed: Optional random seed for reproducibility. When not provided,
///     results may vary between runs.
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
///     communities = rx.leiden_communities(graph)
///     print(communities)
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, max_iterations=100, resolution=None, randomness=None, seed=None))]
pub fn graph_leiden_communities(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    max_iterations: usize,
    resolution: Option<f64>,
    randomness: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_leiden(&weighted_graph, Some(max_iterations), resolution, randomness, seed);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Leiden community detection algorithm for directed graphs.
///
/// This is a hierarchical, modularity-based community detection algorithm
/// that improves upon Louvain by guaranteeing well-connected communities.
/// It works in three phases per iteration:
/// 1. **Local moving**: Each node is moved to the community of a neighbor if it
///    improves the modularity. This is repeated until no improvement is possible.
/// 2. **Refinement**: Each community is split into well-connected sub-communities,
///    ensuring the final communities are connected.
/// 3. **Aggregation**: The refined communities form a new aggregated graph, and
///    the process repeats.
///
/// The algorithm terminates when no further modularity improvement is possible.
///
/// The algorithm is described in:
/// Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
/// guaranteeing well-connected communities. Scientific Reports, 9(1), 5233.
///
/// :param graph: The input graph (PyDiGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param max_iterations: Maximum number of iterations to perform.
///     Default: 100.
/// :param resolution: Resolution parameter (gamma). Values > 1 produce more
///     communities, values < 1 produce fewer. Default: 1.0.
/// :param seed: Optional random seed for reproducibility. When not provided,
///     results may vary between runs.
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
///     communities = rx.digraph_leiden_communities(graph)
///     print(communities)
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, max_iterations=100, resolution=None, randomness=None, seed=None))]
pub fn digraph_leiden_communities(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    max_iterations: usize,
    resolution: Option<f64>,
    randomness: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_leiden(&weighted_graph, Some(max_iterations), resolution, randomness, seed);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Girvan-Newman community detection algorithm.
///
/// This algorithm detects communities by iteratively removing edges with the
/// highest betweenness centrality. At each step, the connected components of
/// the remaining graph define the community partition.
///
/// The algorithm is described in:
/// Girvan, M., & Newman, M. E. J. (2002). Community structure in social
/// and biological networks. Proceedings of the National Academy of Sciences,
/// 99(12), 7821-7826.
///
/// :param graph: The input graph (PyGraph or PyDiGraph) to analyze.
/// :param max_steps: Maximum number of edges to remove. Each step removes
///     the edge with highest betweenness centrality and records the resulting
///     partition. If not provided, all edges will be removed.
///
/// :returns: A list of dictionaries, where each dictionary represents the
///     community partition at that step. The first element (index 0) is the
///     initial partition (all nodes in community 0), and each subsequent
///     element shows the partition after one more edge removal.
///     Community labels are integers starting from 0 at each step.
/// :rtype: list[dict]
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///
///     # Create a graph with two clear communities
///     graph = rx.PyGraph()
///     # Community 1: triangle
///     a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
///     graph.add_edge(a, b, 1.0)
///     graph.add_edge(b, c, 1.0)
///     graph.add_edge(a, c, 1.0)
///     # Community 2: triangle
///     d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
///     graph.add_edge(d, e, 1.0)
///     graph.add_edge(e, f, 1.0)
///     graph.add_edge(d, f, 1.0)
///     # Bridge between communities
///     graph.add_edge(c, d, 1.0)
///
///     dendrogram = rx.girvan_newman(graph, max_steps=1)
///     print(f"Initial partition: {dendrogram[0]}")
///     print(f"After 1 removal: {dendrogram[1]}")
///
#[pyfunction]
#[pyo3(signature = (graph, /, max_steps=None))]
pub fn graph_girvan_newman(
    py: Python,
    graph: &graph::PyGraph,
    max_steps: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &None, 1.0)?;
    let result = core_girvan_newman(&weighted_graph, max_steps);

    let out_list = pyo3::types::PyList::empty(py);
    for partition in result {
        let part_dict = PyDict::new(py);
        for (node, comm) in partition {
            part_dict.set_item(graph.graph.to_index(node), comm)?;
        }
        out_list.append(part_dict)?;
    }
    Ok(out_list.into())
}

/// Girvan-Newman community detection algorithm for directed graphs.
///
/// This algorithm detects communities by iteratively removing edges with the
/// highest betweenness centrality. At each step, the connected components of
/// the remaining graph define the community partition.
///
/// The algorithm is described in:
/// Girvan, M., & Newman, M. E. J. (2002). Community structure in social
/// and biological networks. Proceedings of the National Academy of Sciences,
/// 99(12), 7821-7826.
///
/// :param graph: The input graph (PyDiGraph) to analyze.
/// :param max_steps: Maximum number of edges to remove. Each step removes
///     the edge with highest betweenness centrality and records the resulting
///     partition. If not provided, all edges will be removed.
///
/// :returns: A list of dictionaries, where each dictionary represents the
///     community partition at that step. The first element (index 0) is the
///     initial partition (all nodes in community 0), and each subsequent
///     element shows the partition after one more edge removal.
///     Community labels are integers starting from 0 at each step.
/// :rtype: list[dict]
///
#[pyfunction]
#[pyo3(signature = (graph, /, max_steps=None))]
pub fn digraph_girvan_newman(
    py: Python,
    graph: &digraph::PyDiGraph,
    max_steps: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &None, 1.0)?;
    let result = core_girvan_newman(&weighted_graph, max_steps);

    let out_list = pyo3::types::PyList::empty(py);
    for partition in result {
        let part_dict = PyDict::new(py);
        for (node, comm) in partition {
            part_dict.set_item(graph.graph.to_index(node), comm)?;
        }
        out_list.append(part_dict)?;
    }
    Ok(out_list.into())
}

/// Greedy modularity optimization community detection algorithm.
///
/// This algorithm detects communities by iteratively merging pairs of
/// communities that produce the largest increase in modularity. It starts
/// with each node in its own community (the Clauset-Newman-Moore algorithm).
///
/// The algorithm is described in:
/// Clauset, A., Newman, M. E. J., & Moore, C. (2004). Finding community
/// structure in very large networks. Physical Review E, 70(6), 066111.
///
/// :param graph: The input graph (PyGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
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
///     communities = rx.greedy_modularity_communities(graph)
///     print(communities)
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, resolution=None))]
pub fn graph_greedy_modularity_communities(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    resolution: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_greedy_modularity(&weighted_graph, resolution);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Infomap community detection algorithm.
///
/// This algorithm detects communities by minimizing the map equation,
/// an information-theoretic objective that finds the partition that
/// best compresses the description of random walks on the graph.
///
/// The algorithm is described in:
/// Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks
/// on complex networks reveal community structure.
/// Proceedings of the National Academy of Sciences, 105(4), 1118-1123.
///
/// :param graph: The input graph (PyGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param max_iterations: Maximum number of iterations to perform.
///     Default: 100.
/// :param teleport_prob: Teleportation probability for the random walk
///     (PageRank-style damping). Default: 0.15.
/// :param seed: Optional random seed for reproducibility.
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
///     # Community 1
///     a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
///     graph.add_edge(a, b, 1.0)
///     graph.add_edge(b, c, 1.0)
///     graph.add_edge(a, c, 1.0)
///     # Community 2
///     d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
///     graph.add_edge(d, e, 1.0)
///     graph.add_edge(e, f, 1.0)
///     graph.add_edge(d, f, 1.0)
///     # Bridge
///     graph.add_edge(c, d, 1.0)
///
///     communities = rx.infomap_communities(graph)
///     print(communities)
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, max_iterations=None, teleport_prob=None, seed=None))]
pub fn graph_infomap_communities(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    max_iterations: Option<usize>,
    teleport_prob: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_infomap(&weighted_graph, max_iterations, teleport_prob, seed);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Greedy modularity optimization community detection algorithm for directed graphs.
///
/// This algorithm detects communities by iteratively merging pairs of
/// communities that produce the largest increase in modularity. It starts
/// with each node in its own community (the Clauset-Newman-Moore algorithm).
///
/// The algorithm is described in:
/// Clauset, A., Newman, M. E. J., & Moore, C. (2004). Finding community
/// structure in very large networks. Physical Review E, 70(6), 066111.
///
/// :param graph: The input graph (PyDiGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param resolution: Resolution parameter (gamma). Values > 1 produce more
///     communities, values < 1 produce fewer. Default: 1.0.
///
/// :returns: A dictionary mapping node indices to community labels (integers).
///     Nodes in the same community share the same label.
/// :rtype: dict
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, resolution=None))]
pub fn digraph_greedy_modularity_communities(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    resolution: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_greedy_modularity(&weighted_graph, resolution);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Infomap community detection algorithm for directed graphs.
///
/// This algorithm detects communities by minimizing the map equation,
/// an information-theoretic objective that finds the partition that
/// best compresses the description of random walks on the graph.
///
/// The algorithm is described in:
/// Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks
/// on complex networks reveal community structure.
/// Proceedings of the National Academy of Sciences, 105(4), 1118-1123.
///
/// :param graph: The input graph (PyDiGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param max_iterations: Maximum number of iterations to perform.
///     Default: 100.
/// :param teleport_prob: Teleportation probability for the random walk
///     (PageRank-style damping). Default: 0.15.
/// :param seed: Optional random seed for reproducibility.
///
/// :returns: A dictionary mapping node indices to community labels (integers).
///     Nodes in the same community share the same label.
/// :rtype: dict
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, max_iterations=None, teleport_prob=None, seed=None))]
pub fn digraph_infomap_communities(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    max_iterations: Option<usize>,
    teleport_prob: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_infomap(&weighted_graph, max_iterations, teleport_prob, seed);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Walktrap community detection algorithm for undirected graphs.
///
/// This algorithm detects communities by using short random walks to compute
/// distances between nodes, then performing agglomerative hierarchical clustering.
/// The key insight is that random walks tend to stay within communities, so
/// nodes in the same community have similar random walk probability distributions.
///
/// The algorithm is described in:
/// Pons, P., & Latapy, M. (2005). Computing communities in large networks
/// using random walks. Journal of Graph Algorithms and Applications, 10(2), 191-218.
///
/// :param graph: The input graph (PyGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param walk_length: Length of random walks for computing distances.
///     Default: 4.
/// :param seed: Optional random seed for reproducibility. Reserved for future
///     use; currently unused since the algorithm is deterministic.
///
/// :returns: A dictionary mapping node indices to community labels (integers).
///     Nodes in the same community share the same label. Labels are normalized
///     to compact integers starting from 0.
/// :rtype: dict
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, walk_length=None, seed=None))]
pub fn graph_walktrap_communities(
    py: Python,
    graph: &graph::PyGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    walk_length: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_walktrap(&weighted_graph, walk_length, seed);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Walktrap community detection algorithm for directed graphs.
///
/// This algorithm detects communities by using short random walks to compute
/// distances between nodes, then performing agglomerative hierarchical clustering.
///
/// :param graph: The input graph (PyDiGraph) to analyze. Edge weights must be
///     floating point numbers.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param walk_length: Length of random walks for computing distances.
///     Default: 4.
/// :param seed: Optional random seed for reproducibility. Reserved for future
///     use; currently unused since the algorithm is deterministic.
///
/// :returns: A dictionary mapping node indices to community labels (integers).
///     Nodes in the same community share the same label.
/// :rtype: dict
///
#[pyfunction]
#[pyo3(signature = (graph, /, weight_fn=None, default_weight=1.0, walk_length=None, seed=None))]
pub fn digraph_walktrap_communities(
    py: Python,
    graph: &digraph::PyDiGraph,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    walk_length: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let result = core_walktrap(&weighted_graph, walk_length, seed);

    let out_dict = PyDict::new(py);
    for (node, label) in result {
        out_dict.set_item(graph.graph.to_index(node), label)?;
    }
    Ok(out_dict.into())
}

/// Compute the modularity of a partition.
///
/// Modularity is a measure of the quality of a community partition.
/// It compares the density of edges within communities to the expected
/// density if edges were distributed randomly.
///
/// :param graph: The input graph (PyGraph or PyDiGraph) to analyze.
/// :param communities: A dictionary mapping node indices to community labels.
/// :param resolution: Resolution parameter (gamma). Values > 1 produce more
///     communities, values < 1 produce fewer. Default: 1.0.
///
/// :returns: The modularity score as a float. Higher values indicate better
///     community structure. The maximum possible value is 1.0.
/// :rtype: float
///
/// .. jupyter-execute::
///
///     import rustworkx as rx
///
///     graph = rx.PyGraph()
///     a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
///     d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
///     graph.add_edge(a, b, 1.0)
///     graph.add_edge(b, c, 1.0)
///     graph.add_edge(a, c, 1.0)
///     graph.add_edge(d, e, 1.0)
///     graph.add_edge(e, f, 1.0)
///     graph.add_edge(d, f, 1.0)
///
///     communities = {a: 0, b: 0, c: 0, d: 1, e: 1, f: 1}
///     q = rx.graph_modularity(graph, communities)
///     print(f"Modularity: {q:.4f}")
///
#[pyfunction]
#[pyo3(signature = (graph, communities, /, weight_fn=None, default_weight=1.0, resolution=None))]
pub fn graph_modularity(
    py: Python,
    graph: &graph::PyGraph,
    communities: &Bound<'_, PyDict>,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    resolution: Option<f64>,
) -> PyResult<f64> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let mut comm_map: DictMap<NodeIndex, u32> = DictMap::new();
    for (key, value) in communities {
        let node_idx = key.extract::<usize>()?;
        let label = value.extract::<u32>()?;
        comm_map.insert(NodeIndex::new(node_idx), label);
    }
    let result = rustworkx_core::community::modularity(&weighted_graph, &comm_map, resolution)
        .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
    Ok(result)
}

/// Compute the modularity of a partition for a directed graph.
///
/// :param graph: The input graph (PyDiGraph) to analyze.
/// :param communities: A dictionary mapping node indices to community labels.
/// :param weight_fn: Optional callable to extract edge weights. Takes an edge
///     data object and returns a float. If not provided, uses default_weight.
/// :param default_weight: Default edge weight used when weight_fn is not
///     provided. Default: 1.0.
/// :param resolution: Resolution parameter (gamma). Values > 1 produce more
///     communities, values < 1 produce fewer. Default: 1.0.
///
/// :returns: The modularity score as a float.
/// :rtype: float
#[pyfunction]
#[pyo3(signature = (graph, communities, /, weight_fn=None, default_weight=1.0, resolution=None))]
pub fn digraph_modularity(
    py: Python,
    graph: &digraph::PyDiGraph,
    communities: &Bound<'_, PyDict>,
    weight_fn: Option<Py<PyAny>>,
    default_weight: f64,
    resolution: Option<f64>,
) -> PyResult<f64> {
    let weighted_graph = build_f64_graph(py, &graph.graph, &weight_fn, default_weight)?;
    let mut comm_map: DictMap<NodeIndex, u32> = DictMap::new();
    for (key, value) in communities {
        let node_idx = key.extract::<usize>()?;
        let label = value.extract::<u32>()?;
        comm_map.insert(NodeIndex::new(node_idx), label);
    }
    let result = rustworkx_core::community::modularity(&weighted_graph, &comm_map, resolution)
        .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))?;
    Ok(result)
}
