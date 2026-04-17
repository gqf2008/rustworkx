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

//! Community detection algorithms for graphs.
//!
//! This module provides algorithms for detecting communities (clusters)
//! in graphs. Communities are groups of nodes that are more densely
//! connected internally than to the rest of the graph.

mod girvan_newman;
mod greedy_modularity;
mod infomap;
mod label_propagation;
mod leiden;
mod louvain;
mod modularity;
mod walktrap;

pub use girvan_newman::girvan_newman;
pub use greedy_modularity::greedy_modularity_communities;
pub use infomap::infomap_communities;
pub use label_propagation::label_propagation;
pub use leiden::leiden_communities;
pub use louvain::louvain_communities;
pub use modularity::modularity;
pub use walktrap::walktrap_communities;
