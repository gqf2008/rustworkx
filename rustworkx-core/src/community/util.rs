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

//! Shared utilities for community detection algorithms.

use hashbrown::HashMap;

use crate::dictmap::{DictMap, InitWithHasher};

/// Normalize raw community labels to compact integers starting from 0.
///
/// `raw_labels[i]` is the community label for `nodes[i]`. The returned
/// `DictMap` maps each node to a compact label where the first distinct
/// raw label gets 0, the next new distinct label gets 1, etc.
pub(crate) fn normalize_labels<N>(
    nodes: &[N],
    raw_labels: &[u32],
) -> DictMap<N, u32>
where
    N: Eq + std::hash::Hash + Copy,
{
    debug_assert_eq!(nodes.len(), raw_labels.len());

    let mut label_map: HashMap<u32, u32> = HashMap::new();
    let mut next_label: u32 = 0;
    let mut result = DictMap::with_capacity(nodes.len());

    for (&node, &raw) in nodes.iter().zip(raw_labels.iter()) {
        let compact = *label_map.entry(raw).or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });
        result.insert(node, compact);
    }
    result
}
