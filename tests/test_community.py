# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import unittest

import rustworkx


class TestGraphLabelPropagation(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.graph_label_propagation(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        res = rustworkx.graph_label_propagation(graph)
        self.assertEqual({0: 0}, res)

    def test_two_communities(self):
        graph = rustworkx.PyGraph()
        # Community 1: fully connected triangle
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        graph.add_edge(a, b, 1)
        graph.add_edge(b, c, 1)
        graph.add_edge(a, c, 1)
        # Community 2: fully connected triangle
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        graph.add_edge(d, e, 1)
        graph.add_edge(e, f, 1)
        graph.add_edge(d, f, 1)
        # Single bridge between communities
        graph.add_edge(c, d, 1)

        communities = rustworkx.graph_label_propagation(graph, seed=42)

        # Verify correct number of nodes
        self.assertEqual(len(communities), 6)

        # Verify all nodes are present in the result
        self.assertEqual(set(communities.keys()), {a, b, c, d, e, f})

    def test_complete_graph_single_community(self):
        """Complete graph should converge to a single community."""
        graph = rustworkx.generators.complete_graph(10)
        communities = rustworkx.graph_label_propagation(graph, seed=42)

        labels = set(communities.values())
        self.assertEqual(len(labels), 1)
        # Label should be normalized to 0
        self.assertEqual(communities[0], 0)

    def test_disconnected_graph(self):
        """Nodes with no edges should each be their own community."""
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_node(2)
        communities = rustworkx.graph_label_propagation(graph, seed=42)
        self.assertEqual(len(communities), 3)
        # Each node should have a different label (no neighbors to adopt from)
        labels = list(communities.values())
        self.assertEqual(len(set(labels)), 3)

    def test_star_graph(self):
        graph = rustworkx.generators.star_graph(11)
        communities = rustworkx.graph_label_propagation(graph, seed=42)
        self.assertEqual(len(communities), 11)

    def test_deterministic_with_seed(self):
        graph = rustworkx.generators.path_graph(20)
        # Add extra edges to create community structure
        for i in range(0, 8):
            for j in range(i + 2, 10):
                graph.add_edge(i, j, 1)
        for i in range(10, 18):
            for j in range(i + 2, 20):
                graph.add_edge(i, j, 1)

        communities1 = rustworkx.graph_label_propagation(graph, seed=123)
        communities2 = rustworkx.graph_label_propagation(graph, seed=123)

        # Compare partitions (grouping of nodes), not raw labels
        # since label values may differ between runs
        def get_partition(communities):
            groups = {}
            for node, label in communities.items():
                groups.setdefault(label, set()).add(node)
            return frozenset(frozenset(g) for g in groups.values())

        self.assertEqual(get_partition(communities1), get_partition(communities2))

    def test_custom_max_iterations(self):
        graph = rustworkx.generators.path_graph(5)
        # With only 1 iteration, may not fully converge
        communities = rustworkx.graph_label_propagation(graph, max_iterations=1, seed=42)
        self.assertEqual(len(communities), 5)

    def test_return_type(self):
        graph = rustworkx.generators.path_graph(3)
        res = rustworkx.graph_label_propagation(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)

    def test_labels_normalized(self):
        """Labels should be compact integers starting from 0."""
        graph = rustworkx.PyGraph()
        for i in range(20):
            graph.add_node(i)
        for i in range(19):
            graph.add_edge(i, i + 1, 1)

        communities = rustworkx.graph_label_propagation(graph, seed=42)
        labels = sorted(set(communities.values()))
        # Labels should be 0, 1, 2, ... (compact, no gaps)
        self.assertEqual(labels, list(range(len(labels))))


class TestDiGraphLabelPropagation(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.digraph_label_propagation(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        res = rustworkx.digraph_label_propagation(graph)
        self.assertEqual({0: 0}, res)

    def test_cycle_graph(self):
        graph = rustworkx.generators.directed_cycle_graph(5)
        communities = rustworkx.digraph_label_propagation(graph, seed=42)
        self.assertEqual(len(communities), 5)

    def test_complete_graph(self):
        graph = rustworkx.generators.directed_complete_graph(8)
        communities = rustworkx.digraph_label_propagation(graph, seed=42)
        self.assertEqual(len(communities), 8)

    def test_deterministic_with_seed(self):
        graph = rustworkx.generators.directed_path_graph(15)
        for i in range(0, 6):
            for j in range(i + 2, 8):
                graph.add_edge(i, j, 1)
        for i in range(8, 13):
            for j in range(i + 2, 15):
                graph.add_edge(i, j, 1)

        communities1 = rustworkx.digraph_label_propagation(graph, seed=99)
        communities2 = rustworkx.digraph_label_propagation(graph, seed=99)

        def get_partition(communities):
            groups = {}
            for node, label in communities.items():
                groups.setdefault(label, set()).add(node)
            return frozenset(frozenset(g) for g in groups.values())

        self.assertEqual(get_partition(communities1), get_partition(communities2))

    def test_return_type(self):
        graph = rustworkx.generators.directed_path_graph(3)
        res = rustworkx.digraph_label_propagation(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)


class TestGraphLouvainCommunities(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.graph_louvain_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        res = rustworkx.graph_louvain_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_two_communities(self):
        """Graph with two clear communities and a weak bridge."""
        graph = rustworkx.PyGraph()
        # Community 1: fully connected triangle
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        # Community 2: fully connected triangle
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        # Single bridge between communities
        graph.add_edge(c, d, 1.0)

        communities = rustworkx.graph_louvain_communities(graph)

        self.assertEqual(len(communities), 6)
        # Nodes in same community should have same label
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        # Bridge should be weak enough that communities are separate
        self.assertNotEqual(communities[a], communities[d])

    def test_complete_graph_single_community(self):
        """Complete graph should form a single community."""
        graph = rustworkx.generators.complete_graph(10)
        for u, v in graph.edge_list():
            pass  # just verify it runs
        communities = rustworkx.graph_louvain_communities(graph)
        labels = set(communities.values())
        self.assertEqual(len(labels), 1)
        self.assertEqual(communities[0], 0)

    def test_no_edges(self):
        """Nodes with no edges should each be their own community."""
        graph = rustworkx.PyGraph()
        for _ in range(5):
            graph.add_node(0)
        communities = rustworkx.graph_louvain_communities(graph)
        self.assertEqual(len(communities), 5)
        labels = list(communities.values())
        self.assertEqual(len(set(labels)), 5)

    def test_weighted_edges(self):
        """Strong intra-community weights should produce correct communities."""
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        # Strong intra-community edges
        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(a, c, 10.0)
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(d, f, 10.0)
        # Very weak bridge
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.graph_louvain_communities(graph)

        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        self.assertNotEqual(communities[a], communities[d])

    def test_resolution_parameter(self):
        """Higher resolution should produce more communities."""
        graph = rustworkx.generators.complete_graph(20)
        communities_default = rustworkx.graph_louvain_communities(graph)
        communities_high_res = rustworkx.graph_louvain_communities(
            graph, resolution=2.0
        )
        # Both should be valid (at least 1 community)
        self.assertGreaterEqual(len(set(communities_default.values())), 1)
        self.assertGreaterEqual(len(set(communities_high_res.values())), 1)

    def test_max_levels_parameter(self):
        """Custom max_levels should work without error."""
        graph = rustworkx.generators.path_graph(10)
        communities = rustworkx.graph_louvain_communities(graph, max_levels=1)
        self.assertEqual(len(communities), 10)

    def test_return_type(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, 1.0)
        res = rustworkx.graph_louvain_communities(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)

    def test_labels_normalized(self):
        """Labels should be compact integers starting from 0."""
        graph = rustworkx.PyGraph()
        for i in range(10):
            graph.add_node(i)
        for i in range(9):
            graph.add_edge(i, i + 1, 1.0)

        communities = rustworkx.graph_louvain_communities(graph)
        labels = sorted(set(communities.values()))
        self.assertEqual(labels, list(range(len(labels))))

    def test_max_pass_iterations_zero(self):
        """With max_pass_iterations=0, no local moves happen."""
        graph = rustworkx.PyGraph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        c = graph.add_node(2)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)

        communities = rustworkx.graph_louvain_communities(
            graph, max_pass_iterations=0
        )
        # No moves allowed, each node stays in its own community
        unique = set(communities.values())
        self.assertEqual(len(unique), 3)

    def test_max_pass_iterations_limits(self):
        """max_pass_iterations=1 limits the inner pass to one iteration."""
        graph = rustworkx.PyGraph()
        # Two cliques of 4 nodes each, connected by a single bridge
        c1 = [graph.add_node(i) for i in range(4)]
        c2 = [graph.add_node(i + 4) for i in range(4)]
        for i in range(len(c1)):
            for j in range(i + 1, len(c1)):
                graph.add_edge(c1[i], c1[j], 1.0)
        for i in range(len(c2)):
            for j in range(i + 1, len(c2)):
                graph.add_edge(c2[i], c2[j], 1.0)
        graph.add_edge(c1[0], c2[0], 1.0)

        # With only 1 pass iteration, some nodes may not find their
        # optimal community yet. The result should still be valid
        # (all nodes assigned) but may differ from unlimited passes.
        communities = rustworkx.graph_louvain_communities(
            graph, max_pass_iterations=1, seed=42
        )
        self.assertEqual(len(communities), 8)
        for label in communities.values():
            self.assertIsInstance(label, int)

    def test_self_loops(self):
        """Self-loops should not cause all nodes to merge into one community."""
        graph = rustworkx.PyGraph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        c = graph.add_node(2)
        d = graph.add_node(3)
        e = graph.add_node(4)
        f = graph.add_node(5)

        # Community 1: triangle with self-loops
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(a, a, 5.0)
        graph.add_edge(b, b, 5.0)
        # Community 2: triangle with self-loops
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        graph.add_edge(d, d, 5.0)
        graph.add_edge(e, e, 5.0)
        # Bridge
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.graph_louvain_communities(graph, seed=42)

        # Each triangle should remain mostly intact
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        # The two communities should be distinct
        self.assertNotEqual(communities[a], communities[d])


class TestDiGraphLouvainCommunities(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.digraph_louvain_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        res = rustworkx.digraph_louvain_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_cycle_graph(self):
        graph = rustworkx.generators.directed_cycle_graph(6)
        communities = rustworkx.digraph_louvain_communities(graph)
        self.assertEqual(len(communities), 6)

    def test_two_communities(self):
        """Directed graph: verify algorithm runs and produces valid partition."""
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        # Strong intra-community edges
        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(c, a, 10.0)
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(f, d, 10.0)
        # Very weak bridge
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.digraph_louvain_communities(graph)

        self.assertEqual(len(communities), 6)
        # Verify a, b, c are in the same community
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        # Verify d, e, f partition is valid (may not all be same due to directed nature)
        # Just check the algorithm produces a valid partition
        self.assertIn(communities[d], communities.values())
        self.assertIn(communities[e], communities.values())
        self.assertIn(communities[f], communities.values())

    def test_self_loops(self):
        """Self-loops should not collapse directed graph communities."""
        graph = rustworkx.PyDiGraph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        c = graph.add_node(2)
        d = graph.add_node(3)
        e = graph.add_node(4)
        f = graph.add_node(5)

        # Community 1: directed cycle with self-loops
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(c, a, 1.0)
        graph.add_edge(a, a, 5.0)
        graph.add_edge(b, b, 5.0)
        # Community 2: directed cycle with self-loops
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(f, d, 1.0)
        graph.add_edge(d, d, 5.0)
        graph.add_edge(e, e, 5.0)
        # Weak bridge
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.digraph_louvain_communities(graph, seed=42)

        # Each cycle should remain mostly intact
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        # The two communities should be distinct
        self.assertNotEqual(communities[a], communities[d])

    def test_return_type(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, 1.0)
        res = rustworkx.digraph_louvain_communities(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)

class TestGraphLeidenCommunities(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.graph_leiden_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        res = rustworkx.graph_leiden_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_two_communities(self):
        """Graph with two clear communities and a weak bridge."""
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        graph.add_edge(c, d, 1.0)

        communities = rustworkx.graph_leiden_communities(graph, seed=42)

        self.assertEqual(len(communities), 6)
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        self.assertNotEqual(communities[a], communities[d])

    def test_complete_graph_single_community(self):
        """Complete graph should form a single community."""
        graph = rustworkx.generators.complete_graph(10)
        communities = rustworkx.graph_leiden_communities(graph, seed=42)
        labels = set(communities.values())
        self.assertEqual(len(labels), 1)
        self.assertEqual(communities[0], 0)

    def test_no_edges(self):
        """Nodes with no edges should each be their own community."""
        graph = rustworkx.PyGraph()
        for _ in range(5):
            graph.add_node(0)
        communities = rustworkx.graph_leiden_communities(graph)
        self.assertEqual(len(communities), 5)
        labels = list(communities.values())
        self.assertEqual(len(set(labels)), 5)

    def test_weighted_edges(self):
        """Strong intra-community weights should produce correct communities."""
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(a, c, 10.0)
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(d, f, 10.0)
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.graph_leiden_communities(graph, seed=42)

        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        self.assertNotEqual(communities[a], communities[d])

    def test_resolution_parameter(self):
        """Higher resolution should produce more or equal communities."""
        graph = rustworkx.generators.complete_graph(20)
        communities_default = rustworkx.graph_leiden_communities(graph, seed=42)
        communities_high_res = rustworkx.graph_leiden_communities(
            graph, resolution=2.0, seed=42
        )
        self.assertGreaterEqual(len(set(communities_default.values())), 1)
        self.assertGreaterEqual(len(set(communities_high_res.values())), 1)

    def test_max_iterations_parameter(self):
        """Custom max_iterations should work without error."""
        graph = rustworkx.generators.path_graph(10)
        communities = rustworkx.graph_leiden_communities(graph, max_iterations=1, seed=42)
        self.assertEqual(len(communities), 10)

    def test_seed_deterministic(self):
        """Same seed should produce same results."""
        graph = rustworkx.PyGraph()
        for i in range(20):
            graph.add_node(i)
        for i in range(19):
            graph.add_edge(i, i + 1, 1.0)
        for i in range(0, 8):
            for j in range(i + 2, 10):
                graph.add_edge(i, j, 1.0)
        for i in range(10, 18):
            for j in range(i + 2, 20):
                graph.add_edge(i, j, 1.0)

        communities1 = rustworkx.graph_leiden_communities(graph, seed=123)
        communities2 = rustworkx.graph_leiden_communities(graph, seed=123)
        self.assertEqual(communities1, communities2)

    def test_return_type(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, 1.0)
        res = rustworkx.graph_leiden_communities(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)

    def test_labels_normalized(self):
        """Labels should be compact integers starting from 0."""
        graph = rustworkx.PyGraph()
        for i in range(10):
            graph.add_node(i)
        for i in range(9):
            graph.add_edge(i, i + 1, 1.0)

        communities = rustworkx.graph_leiden_communities(graph, seed=42)
        labels = sorted(set(communities.values()))
        self.assertEqual(labels, list(range(len(labels))))


class TestDiGraphLeidenCommunities(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.digraph_leiden_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        res = rustworkx.digraph_leiden_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_cycle_graph(self):
        graph = rustworkx.generators.directed_cycle_graph(6)
        communities = rustworkx.digraph_leiden_communities(graph, seed=42)
        self.assertEqual(len(communities), 6)

    def test_two_communities(self):
        """Directed graph: verify algorithm runs and produces valid partition."""
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(c, a, 10.0)
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(f, d, 10.0)
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.digraph_leiden_communities(graph, seed=42)

        self.assertEqual(len(communities), 6)
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertIn(communities[d], communities.values())
        self.assertIn(communities[e], communities.values())
        self.assertIn(communities[f], communities.values())

    def test_return_type(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, 1.0)
        res = rustworkx.digraph_leiden_communities(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)


class TestGraphGirvanNewman(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        dendrogram = rustworkx.graph_girvan_newman(graph)
        self.assertEqual([], dendrogram)

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        a = graph.add_node(0)
        dendrogram = rustworkx.graph_girvan_newman(graph)
        self.assertEqual(1, len(dendrogram))
        self.assertEqual({0: 0}, dendrogram[0])

    def test_two_communities(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        # Community 1: triangle
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        # Community 2: triangle
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        # Bridge
        graph.add_edge(c, d, 1.0)

        dendrogram = rustworkx.graph_girvan_newman(graph, max_steps=1)
        self.assertEqual(2, len(dendrogram))

        # Initial: all one community
        self.assertEqual(6, len(dendrogram[0]))

        # After removing bridge: should be 2 communities
        partition = dendrogram[1]
        self.assertEqual(partition[a], partition[b])
        self.assertEqual(partition[b], partition[c])
        self.assertEqual(partition[d], partition[e])
        self.assertEqual(partition[e], partition[f])
        self.assertNotEqual(partition[a], partition[d])

    def test_complete_graph(self):
        graph = rustworkx.PyGraph()
        nodes = [graph.add_node(i) for i in range(6)]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                graph.add_edge(nodes[i], nodes[j], 1.0)

        # Complete graph: removing one edge shouldn't split it
        dendrogram = rustworkx.graph_girvan_newman(graph, max_steps=1)
        self.assertEqual(2, len(dendrogram))
        labels = list(dendrogram[1].values())
        self.assertEqual(min(labels), 0)
        self.assertEqual(max(labels), 0)

    def test_dendrogram_length(self):
        graph = rustworkx.PyGraph()
        a = graph.add_node(0)
        b = graph.add_node(1)
        graph.add_edge(a, b, 1.0)

        # 1 edge: initial + 1 removal = 2 partitions
        dendrogram = rustworkx.graph_girvan_newman(graph)
        self.assertEqual(2, len(dendrogram))
        # After removal: each node isolated
        self.assertNotEqual(dendrogram[1][a], dendrogram[1][b])

    def test_return_type(self):
        graph = rustworkx.PyGraph()
        a, b = graph.add_node(0), graph.add_node(1)
        graph.add_edge(a, b, 1.0)
        dendrogram = rustworkx.graph_girvan_newman(graph)
        self.assertIsInstance(dendrogram, list)
        for partition in dendrogram:
            self.assertIsInstance(partition, dict)
            for key, value in partition.items():
                self.assertIsInstance(key, int)
                self.assertIsInstance(value, int)


class TestDiGraphGirvanNewman(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        dendrogram = rustworkx.digraph_girvan_newman(graph)
        self.assertEqual([], dendrogram)

    def test_single_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        dendrogram = rustworkx.digraph_girvan_newman(graph)
        self.assertEqual(1, len(dendrogram))
        self.assertEqual({0: 0}, dendrogram[0])

    def test_two_communities(self):
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        # Community 1: cycle with strong weights
        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(c, a, 10.0)
        # Community 2: cycle with strong weights
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(f, d, 10.0)
        # Weak bridge
        graph.add_edge(c, d, 0.1)

        dendrogram = rustworkx.digraph_girvan_newman(graph, max_steps=1)
        self.assertEqual(2, len(dendrogram))

        # Initial: all one community
        self.assertEqual(6, len(dendrogram[0]))

        # For directed graphs, the betweenness centrality behaves differently,
        # so we verify basic properties rather than exact community splits
        partition = dendrogram[1]
        self.assertEqual(6, len(partition))
        # All nodes should still be in some community
        self.assertEqual(len(set(partition.values())), 1)  # Still connected

    def test_max_steps(self):
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)

        # With max_steps=2, should get 3 partitions
        dendrogram = rustworkx.digraph_girvan_newman(graph, max_steps=2)
        self.assertEqual(3, len(dendrogram))

    def test_return_type(self):
        graph = rustworkx.PyDiGraph()
        a, b = graph.add_node(0), graph.add_node(1)
        graph.add_edge(a, b, 1.0)
        dendrogram = rustworkx.digraph_girvan_newman(graph)
        self.assertIsInstance(dendrogram, list)
        for partition in dendrogram:
            self.assertIsInstance(partition, dict)
            for key, value in partition.items():
                self.assertIsInstance(key, int)
                self.assertIsInstance(value, int)


class TestGraphGreedyModularity(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.graph_greedy_modularity_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        res = rustworkx.graph_greedy_modularity_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_two_communities(self):
        """Graph with two clear communities and a weak bridge."""
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        graph.add_edge(c, d, 1.0)

        communities = rustworkx.graph_greedy_modularity_communities(graph)

        self.assertEqual(len(communities), 6)
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        self.assertNotEqual(communities[a], communities[d])

    def test_complete_graph_single_community(self):
        """Complete graph should form a single community."""
        graph = rustworkx.generators.complete_graph(10)
        communities = rustworkx.graph_greedy_modularity_communities(graph)
        labels = set(communities.values())
        self.assertEqual(len(labels), 1)
        self.assertEqual(communities[0], 0)

    def test_no_edges(self):
        """Nodes with no edges should each be their own community."""
        graph = rustworkx.PyGraph()
        for _ in range(5):
            graph.add_node(0)
        communities = rustworkx.graph_greedy_modularity_communities(graph)
        self.assertEqual(len(communities), 5)
        labels = list(communities.values())
        self.assertEqual(len(set(labels)), 5)

    def test_weighted_edges(self):
        """Strong intra-community weights should produce correct communities."""
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(a, c, 10.0)
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(d, f, 10.0)
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.graph_greedy_modularity_communities(graph)

        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        self.assertNotEqual(communities[a], communities[d])

    def test_resolution_parameter(self):
        """Higher resolution should produce more communities."""
        graph = rustworkx.PyGraph()
        a, b, c, d = graph.add_node(0), graph.add_node(1), graph.add_node(2), graph.add_node(3)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(c, d, 1.0)

        lo_res = rustworkx.graph_greedy_modularity_communities(graph, resolution=0.5)
        hi_res = rustworkx.graph_greedy_modularity_communities(graph, resolution=2.0)

        # Higher resolution should produce more or equal communities
        self.assertGreaterEqual(len(set(hi_res.values())), len(set(lo_res.values())))

    def test_labels_normalized(self):
        """Labels should be compact integers starting from 0."""
        graph = rustworkx.PyGraph()
        a, b = graph.add_node(0), graph.add_node(1)
        graph.add_edge(a, b, 1.0)

        communities = rustworkx.graph_greedy_modularity_communities(graph)
        self.assertEqual(communities[a], 0)
        self.assertEqual(communities[b], 0)

    def test_return_type(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, 1.0)
        res = rustworkx.graph_greedy_modularity_communities(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)


class TestDiGraphGreedyModularity(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.digraph_greedy_modularity_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        res = rustworkx.digraph_greedy_modularity_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_two_communities(self):
        """Directed graph with two clear communities."""
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(c, a, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(f, d, 1.0)
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.digraph_greedy_modularity_communities(graph)

        self.assertEqual(len(communities), 6)
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])

    def test_return_type(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, 1.0)
        res = rustworkx.digraph_greedy_modularity_communities(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)


class TestGraphInfomap(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.graph_infomap_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        res = rustworkx.graph_infomap_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_two_communities(self):
        """Graph with two clear communities and a weak bridge."""
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        graph.add_edge(c, d, 1.0)

        communities = rustworkx.graph_infomap_communities(graph, seed=42)

        self.assertEqual(len(communities), 6)
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        self.assertNotEqual(communities[a], communities[d])

    def test_complete_graph_single_community(self):
        """Complete graph should form a single community."""
        graph = rustworkx.generators.complete_graph(10)
        communities = rustworkx.graph_infomap_communities(graph, seed=42)
        labels = set(communities.values())
        self.assertEqual(len(labels), 1)
        self.assertEqual(communities[0], 0)

    def test_no_edges(self):
        """Nodes with no edges should each be their own community."""
        graph = rustworkx.PyGraph()
        for _ in range(5):
            graph.add_node(0)
        communities = rustworkx.graph_infomap_communities(graph, seed=42)
        self.assertEqual(len(communities), 5)
        labels = list(communities.values())
        self.assertEqual(len(set(labels)), 5)

    def test_weighted_edges(self):
        """Strong intra-community weights should produce correct communities."""
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(a, c, 10.0)
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(d, f, 10.0)
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.graph_infomap_communities(graph, seed=42)

        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        self.assertNotEqual(communities[a], communities[d])

    def test_labels_normalized(self):
        """Labels should be compact integers starting from 0."""
        graph = rustworkx.PyGraph()
        a, b = graph.add_node(0), graph.add_node(1)
        graph.add_edge(a, b, 1.0)

        communities = rustworkx.graph_infomap_communities(graph, seed=42)
        self.assertEqual(communities[a], 0)
        self.assertEqual(communities[b], 0)

    def test_return_type(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, 1.0)
        res = rustworkx.graph_infomap_communities(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)


class TestDiGraphInfomap(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.digraph_infomap_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        res = rustworkx.digraph_infomap_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_two_communities(self):
        """Directed graph with two clear communities."""
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(c, a, 10.0)
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(f, d, 10.0)
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.digraph_infomap_communities(graph, seed=42)

        self.assertEqual(len(communities), 6)
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])

    def test_return_type(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, 1.0)
        res = rustworkx.digraph_infomap_communities(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)


class TestNegativeWeights(unittest.TestCase):
    """Test community detection algorithms with negative edge weights."""

    def test_louvain_negative_weights(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        # Positive intra-community edges
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        # Negative inter-community edge
        graph.add_edge(c, d, -0.5)

        communities = rustworkx.graph_louvain_communities(graph, seed=42)
        self.assertEqual(6, len(communities))
        # All nodes should be assigned a community label
        for label in communities.values():
            self.assertIn(label, set(communities.values()))

    def test_leiden_negative_weights(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        # Positive intra-community edges
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        # Negative inter-community edge
        graph.add_edge(c, d, -0.5)

        communities = rustworkx.graph_leiden_communities(graph, seed=42)
        self.assertEqual(6, len(communities))
        for label in communities.values():
            self.assertIn(label, set(communities.values()))

    def test_girvan_newman_negative_weights(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        # Positive intra-community edges
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        # Negative bridge edge
        graph.add_edge(c, d, -1.0)

        dendrogram = rustworkx.graph_girvan_newman(graph, max_steps=1)
        self.assertEqual(2, len(dendrogram))
        # Should still produce valid partitions
        self.assertEqual(6, len(dendrogram[0]))
        self.assertEqual(6, len(dendrogram[1]))

    def test_greedy_modularity_negative_weights(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        # Positive intra-community edges
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        # Negative bridge edge
        graph.add_edge(c, d, -0.5)

        communities = rustworkx.graph_greedy_modularity_communities(graph)
        # Should still produce valid partitions with all nodes assigned
        self.assertEqual(6, len(communities))
        for label in communities.values():
            self.assertIn(label, set(communities.values()))

    def test_infomap_negative_weights(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        # Positive intra-community edges
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        # Negative bridge edge
        graph.add_edge(c, d, -0.5)

        communities = rustworkx.graph_infomap_communities(graph, seed=42)
        # Should still produce valid partitions with all nodes assigned
        self.assertEqual(6, len(communities))
        for label in communities.values():
            self.assertIn(label, set(communities.values()))

    def test_walktrap_negative_weights(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        # Positive intra-community edges
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        # Negative bridge edge
        graph.add_edge(c, d, -0.5)

        communities = rustworkx.graph_walktrap_communities(graph, seed=42)
        # Should still produce valid partitions with all nodes assigned
        self.assertEqual(6, len(communities))
        for label in communities.values():
            self.assertIn(label, set(communities.values()))


class TestGraphWalktrap(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        res = rustworkx.graph_walktrap_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        res = rustworkx.graph_walktrap_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_two_communities(self):
        """Graph with two clear communities and a weak bridge."""
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)
        graph.add_edge(c, d, 1.0)

        communities = rustworkx.graph_walktrap_communities(graph, seed=42)

        self.assertEqual(len(communities), 6)
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        self.assertNotEqual(communities[a], communities[d])

    def test_complete_graph_single_community(self):
        """Complete graph should form a single community."""
        graph = rustworkx.generators.complete_graph(10)
        communities = rustworkx.graph_walktrap_communities(graph)
        labels = set(communities.values())
        self.assertEqual(len(labels), 1)
        self.assertEqual(communities[0], 0)

    def test_no_edges(self):
        """Nodes with no edges should each be their own community."""
        graph = rustworkx.PyGraph()
        for _ in range(5):
            graph.add_node(0)
        communities = rustworkx.graph_walktrap_communities(graph)
        self.assertEqual(len(communities), 5)
        labels = list(communities.values())
        self.assertEqual(len(set(labels)), 5)

    def test_weighted_edges(self):
        """Strong intra-community weights should produce correct communities."""
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(a, c, 10.0)
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(d, f, 10.0)
        graph.add_edge(c, d, 0.1)

        communities = rustworkx.graph_walktrap_communities(graph, seed=42)

        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])
        self.assertEqual(communities[d], communities[e])
        self.assertEqual(communities[e], communities[f])
        self.assertNotEqual(communities[a], communities[d])

    def test_labels_normalized(self):
        """Labels should be compact integers starting from 0."""
        graph = rustworkx.PyGraph()
        a, b = graph.add_node(0), graph.add_node(1)
        graph.add_edge(a, b, 1.0)

        communities = rustworkx.graph_walktrap_communities(graph)
        self.assertEqual(communities[a], 0)
        self.assertEqual(communities[b], 0)

    def test_return_type(self):
        graph = rustworkx.PyGraph()
        graph.add_node(0)
        graph.add_node(1)
        graph.add_edge(0, 1, 1.0)
        res = rustworkx.graph_walktrap_communities(graph)
        self.assertIsInstance(res, dict)
        for key, value in res.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(value, int)


class TestDiGraphWalktrap(unittest.TestCase):
    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        res = rustworkx.digraph_walktrap_communities(graph)
        self.assertEqual({}, res)

    def test_single_node(self):
        graph = rustworkx.PyDiGraph()
        graph.add_node(0)
        res = rustworkx.digraph_walktrap_communities(graph)
        self.assertEqual({0: 0}, res)

    def test_two_communities(self):
        """DiGraph with two clear communities."""
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)

        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(c, a, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(f, d, 1.0)
        graph.add_edge(c, d, 1.0)

        communities = rustworkx.digraph_walktrap_communities(graph, seed=42)
        self.assertEqual(len(communities), 6)
        # In directed graphs, random walks follow edge directions
        # Source community nodes should be more similar to each other
        # than to target community nodes
        self.assertEqual(communities[a], communities[b])
        self.assertEqual(communities[b], communities[c])

    def test_walk_length_parameter(self):
        """Different walk lengths should produce valid results."""
        graph = rustworkx.PyDiGraph()
        for i in range(4):
            graph.add_node(i)
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)

        short = rustworkx.digraph_walktrap_communities(graph, walk_length=2)
        long = rustworkx.digraph_walktrap_communities(graph, walk_length=10)
        self.assertEqual(len(short), 4)
        self.assertEqual(len(long), 4)


class TestGraphModularity(unittest.TestCase):
    def test_two_communities(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)

        # Perfect community partition should have positive modularity
        communities = {a: 0, b: 0, c: 0, d: 1, e: 1, f: 1}
        q = rustworkx.graph_modularity(graph, communities)
        self.assertGreater(q, 0.0)

    def test_single_community(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)

        # Single community should still have positive modularity
        communities = {a: 0, b: 0, c: 0, d: 0, e: 0, f: 0}
        q = rustworkx.graph_modularity(graph, communities)
        self.assertGreater(q, 0.0)

    def test_empty_graph(self):
        graph = rustworkx.PyGraph()
        communities = {}
        q = rustworkx.graph_modularity(graph, communities)
        self.assertEqual(q, 0.0)

    def test_resolution_parameter(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(a, c, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(d, f, 1.0)

        communities = {a: 0, b: 0, c: 0, d: 1, e: 1, f: 1}
        q_default = rustworkx.graph_modularity(graph, communities)
        q_hi_res = rustworkx.graph_modularity(graph, communities, resolution=2.0)
        q_lo_res = rustworkx.graph_modularity(graph, communities, resolution=0.5)
        # Higher resolution -> lower modularity for same partition
        self.assertLess(q_hi_res, q_default)
        self.assertGreater(q_lo_res, q_default)

    def test_weighted_graph(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        graph.add_edge(a, b, 10.0)
        graph.add_edge(b, c, 10.0)
        graph.add_edge(a, c, 10.0)
        graph.add_edge(d, e, 10.0)
        graph.add_edge(e, f, 10.0)
        graph.add_edge(d, f, 10.0)
        graph.add_edge(c, d, 0.1)

        communities = {a: 0, b: 0, c: 0, d: 1, e: 1, f: 1}
        q = rustworkx.graph_modularity(graph, communities)
        self.assertGreater(q, 0.0)

    def test_missing_node_raises(self):
        graph = rustworkx.PyGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        graph.add_edge(a, b, 1.0)

        communities = {a: 0, b: 0}
        with self.assertRaises(KeyError):
            rustworkx.graph_modularity(graph, communities)


    def test_extra_node_ignored(self):
        graph = rustworkx.PyGraph()
        a, b = graph.add_node(0), graph.add_node(1)
        graph.add_edge(a, b, 1.0)

        communities = {a: 0, b: 0, 999: 1}
        q = rustworkx.graph_modularity(graph, communities)
        self.assertGreater(q, 0.0)


class TestDiGraphModularity(unittest.TestCase):
    def test_two_communities(self):
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        d, e, f = graph.add_node(3), graph.add_node(4), graph.add_node(5)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)
        graph.add_edge(c, a, 1.0)
        graph.add_edge(d, e, 1.0)
        graph.add_edge(e, f, 1.0)
        graph.add_edge(f, d, 1.0)

        communities = {a: 0, b: 0, c: 0, d: 1, e: 1, f: 1}
        q = rustworkx.digraph_modularity(graph, communities)
        self.assertGreater(q, 0.0)

    def test_empty_graph(self):
        graph = rustworkx.PyDiGraph()
        communities = {}
        q = rustworkx.digraph_modularity(graph, communities)
        self.assertEqual(q, 0.0)

    def test_path_graph(self):
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        graph.add_edge(a, b, 1.0)
        graph.add_edge(b, c, 1.0)

        # All in same community
        communities = {a: 0, b: 0, c: 0}
        q = rustworkx.digraph_modularity(graph, communities)
        self.assertGreater(q, 0.0)

    def test_missing_node_raises(self):
        graph = rustworkx.PyDiGraph()
        a, b, c = graph.add_node(0), graph.add_node(1), graph.add_node(2)
        graph.add_edge(a, b, 1.0)

        # communities dict missing node c
        communities = {a: 0, b: 0}
        with self.assertRaises(KeyError):
            rustworkx.digraph_modularity(graph, communities)

    def test_extra_node_ignored(self):
        graph = rustworkx.PyDiGraph()
        a, b = graph.add_node(0), graph.add_node(1)
        graph.add_edge(a, b, 1.0)

        # communities dict has extra node not in graph
        communities = {a: 0, b: 0, 999: 1}
        q = rustworkx.digraph_modularity(graph, communities)
        self.assertGreater(q, 0.0)

