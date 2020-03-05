# Artificial Intelligence - Search

## Overview

Search is an integral part of AI. It helps in problem solving across a wide variety of domains where a solution isn’t immediately clear.  Here, we implement several graph search algorithms that will calculate a driving route between two points in Romania with a minimal time and space cost. These algorithms can be generalized to all other map search applications.
1. BFS graph search
2. Uniform Cost search
3. A* graph search algorithm


1. **__map_search_algorithms.py__**: Where  _PriorityQueue_, _Breadth First Search_, _Uniform Cost Search_, _A* Search_, _Bi-directional Search_, Tri-directional Search_ map search algorithms are implemented
2. **_search_submission_tests.py_**: Sample tests to validate your searches locally.
3. **_search_unit_tests.py_**: More detailed tests that run searches from all possible pairs of nodes in the graph
4. **_search_submission_tests_grid.py_**: Tests searches on uniform grid and highlights path and explored nodes.
5. **_romania_graph.pickle_**: Serialized graph files for Romania.
6. **_atlanta_osm.pickle_**: Serialized graph files for Atlanta (optional for robust testing for Race!).
7. **_explorable_graph.py_**: A wrapper around `networkx` that tracks explored nodes. **FOR DEBUGGING ONLY**
9. **_visualize_graph.py_**: Module to visualize search results. See below on how to use it.
10. **_osm2networkx.py_**: Module used by visualize graph to read OSM networks.

### Notes

#### A note on visualizing results for the Atlanta graph:

The Atlanta graph is too big to display within a Python window like Romania. As a result, when you run the bidirectional tests in **_search_submission_tests.py_**, it generates a JSON file in the GeoJSON format. To see the graph, you can upload it to a private GitHub Gist or use [this](http://geojson.io/) site.
If you want to see how **_visualize_graph.py_** is used, take a look at the class TestBidirectionalSearch in **_search_submission_tests.py_**

We will be using an undirected network representing a map of Romania (and an optional Atlanta graph ).
