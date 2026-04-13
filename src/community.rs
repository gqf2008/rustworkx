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

use rustworkx_core::community::label_propagation as core_label_propagation;

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
