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

//! Small deterministic random utilities for community detection algorithms.

/// LCG random number generator (glibc parameters).
pub(crate) fn rand_u64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

/// Uniform random float in [0.0, 1.0) using the LCG above.
pub(crate) fn rand_f64(seed: &mut u64) -> f64 {
    // Use the upper 53 bits for maximum precision in [0,1)
    let r = rand_u64(seed);
    (r >> 11) as f64 / (1u64 << 53) as f64
}

/// In-place Fisher-Yates shuffle using the LCG above.
pub(crate) fn fisher_yates_shuffle<T>(slice: &mut [T], seed: &mut u64) {
    let n = slice.len();
    for i in (1..n).rev() {
        let r = (rand_u64(seed) as usize) % (i + 1);
        slice.swap(i, r);
    }
}
