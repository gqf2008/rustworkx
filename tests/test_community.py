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
