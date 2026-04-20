# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Overview

**rustworkx** is a high-performance, general-purpose graph library for Python, written in Rust. It uses PyO3 to expose Rust code as a Python extension module. Version 0.17.1, Python 3.10–3.14, MSRV Rust 1.85.

## Two-Crate Architecture

The project has two Rust crates:
- **`rustworkx`** (root `Cargo.toml`) — Python extension via PyO3 (`cdylib`). This is the main package users import.
- **`rustworkx-core`** (`rustworkx-core/`) — Pure Rust library with no Python dependency. Contains generic algorithm implementations that the Python extension calls into. Other Rust projects can use this crate independently.

When adding new algorithms, prefer implementing them in `rustworkx-core` first (as generic Rust functions), then expose them in the `rustworkx` Python extension via `#[pyfunction]`.

## Building

```bash
# Install into Python environment (release build)
pip install .

# Develop mode (debug build, slower but debuggable)
SETUPTOOLS_RUST_CARGO_PROFILE=dev pip install .

# Note: `pip install -e` does NOT work — it links the Python shim but doesn't build the Rust binary.
# After any Rust code change, re-run `pip install .` to recompile.
```

## Testing

```bash
# Run all Python tests (via nox + stestr)
nox -e test

# Run tests for specific Python version
nox --python 3.11 -e test_with_version

# Run specific test module
nox -e test -- -n test_max_weight_matching

# Run specific test class
nox -e test -- -n graph.test_nodes.TestNodes

# Run specific test method
nox -e test -- -n graph.test_nodes.TestNodes.test_no_nodes

# Run Rust tests for rustworkx-core
cargo test --workspace

# Run stub type tests
nox -e stubs
```

Tests live in `tests/` and use `stestr` (unittest-based). Test files are organized by component: `tests/digraph/`, `tests/graph/`, `tests/generators/`, `tests/visualization/`.

## Linting & Formatting

```bash
# Run all linting (ruff, cargo fmt, typos, stray release notes)
nox -e lint

# Format Python code only
nox -e format

# Check Rust formatting
cargo fmt --all -- --check

# Run clippy (Rust linter)
cargo clippy -- -D warnings

# Spell check
typos
```

Rust formatter: `max_width = 100` (see `rustfmt.toml`). Python: ruff, line-length 100, target Python 3.10.

## Documentation

```bash
# Build docs (requires uv)
nox -e docs

# Clean docs build
nox -e docs_clean
```

Docs use Sphinx with the Qiskit theme. Source in `docs/source/`.

## Key Source Locations

- **`src/`** — Rust source for Python extension
  - `src/lib.rs` — Main entry point, exports all modules and registers Python functions/classes
  - `src/digraph.rs` — `PyDiGraph` class (~3653 lines)
  - `src/graph.rs` — `PyGraph` class (~2298 lines)
  - `src/shortest_path/` — Shortest path algorithms
  - `src/connectivity/` — Connectivity algorithms
  - `src/traversal/` — Graph traversal
  - `src/centrality.rs` — Centrality algorithms
  - `src/community.rs` — Community detection algorithms (label propagation, louvain)
  - `src/isomorphism/` — Graph isomorphism
  - `src/generators.rs` — Graph generators
  - `src/iterators.rs` — Custom iterator types exposed to Python
- **`rustworkx/`** — Python package
  - `__init__.py` — Re-exports from Rust extension (~101K)
  - `rustworkx.pyi` — Type stubs for Rust extension (update when adding functions)
  - `visualization/` — matplotlib + graphviz visualization modules
- **`rustworkx-core/src/`** — Pure Rust algorithm implementations

## Adding New Functions

1. Implement the algorithm in `rustworkx-core` as a generic Rust function
2. Add a `#[pyfunction]` wrapper in the appropriate `src/` module
3. Register the function in `src/lib.rs` via `wrap_pyfunction!`
4. Add the signature to `rustworkx/rustworkx.pyi`
5. Re-export in `rustworkx/__init__.py`
6. Write tests in `tests/`

## Release Notes

Uses [reno](https://docs.openstack.org/reno/latest/) for git-based release notes:

```bash
# Create a new release note
reno new short-description-string
```

Files go in `releasenotes/notes/`. Run `python tools/find_stray_release_notes.py` to check for orphaned notes.

## Important Notes

- **PyO3 ABI**: Uses `abi3-py310` for stable ABI across Python 3.10+. Functions exposed via `#[pyfunction]`, classes via `#[pyclass]`.
- **Custom exceptions**: Created with PyO3's `create_exception!` macro. See `src/lib.rs` for existing exception types.
- **No Makefile**: All task running goes through `nox`. Common sessions: `test`, `lint`, `format`, `docs`, `stubs`, `typos`.
- **Visualization tests**: Set `RUSTWORKX_TEST_PRESERVE_IMAGES=1` to keep generated images for inspection.
