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

//! Common modularity computation utilities.

use hashbrown::HashMap;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};

use crate::dictmap::DictMap;

/// Error type for modularity computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModularityError {
    /// A node is missing from the communities mapping.
    NodeNotFound,
}

impl std::fmt::Display for ModularityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModularityError::NodeNotFound => write!(f, "node not found in communities"),
        }
    }
}

impl std::error::Error for ModularityError {}

/// Compute the modularity of a given partition.
///
/// Q = (1/2m) * sum[(A_ij - resolution * k_i*k_j/(2m)) * delta(c_i, c_j)]
///
/// The `resolution` parameter defaults to 1.0. Values > 1 produce more communities,
/// values < 1 produce fewer. Use the same resolution as was used in the community
/// detection algorithm for consistent evaluation.
///
/// Returns an error if any node is missing from the communities map.
pub fn modularity<G>(
    graph: G,
    communities: &DictMap<G::NodeId, u32>,
    resolution: Option<f64>,
) -> Result<f64, ModularityError>
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
        return Ok(0.0);
    }

    let mut q = 0.0;
    for node in graph.node_identifiers() {
        let ci = *communities.get(&node).ok_or(ModularityError::NodeNotFound)?;
        let ki = degree[&node];
        for edge in graph.edges(node) {
            let target = edge.target();
            let cj = *communities.get(&target).ok_or(ModularityError::NodeNotFound)?;
            if ci == cj {
                let a_ij: f64 = f64::from(*edge.weight());
                let kj = degree[&target];
                q += a_ij - resolution * (ki * kj) / (2.0 * m);
            }
        }
    }

    Ok(q / (2.0 * m))
}

/// Compute modularity for a single partition from pre-computed arrays.
///
/// `labels[i]` is the community label for node index `i`.
/// `degree[i]` is the sum of edge weights for node `i`.
/// `two_m` is the sum of all degrees (= 2 * total edge weight).
/// `edges` contains all directed edges as (source_idx, target_idx, weight).
/// `resolution` is the resolution parameter gamma.
///
/// For undirected graphs, `edges` should contain each edge in both directions,
/// and `two_m` should be the sum of all directed edge weights.
pub(crate) fn modularity_from_partition(
    labels: &[u32],
    degree: &[f64],
    two_m: f64,
    edges: &[(usize, usize, f64)],
    resolution: f64,
) -> f64 {
    if two_m == 0.0 {
        return 0.0;
    }

    let max_comm = labels.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut sum_deg = vec![0.0; max_comm];
    for i in 0..labels.len() {
        sum_deg[labels[i] as usize] += degree[i];
    }

    let expected: f64 = sum_deg.iter().map(|d| d * d).sum::<f64>() / (2.0 * two_m);

    let mut actual = 0.0;
    for &(i, j, w) in edges {
        if labels[i] == labels[j] {
            actual += w;
        }
    }
    // For undirected graphs, edges are provided in both directions,
    // so actual counts each undirected edge twice.
    actual /= 2.0;

    (actual - resolution * expected) / (two_m / 2.0)
}
