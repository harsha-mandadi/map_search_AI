# coding=utf-8

import heapq
import os
import pickle
import math
import copy


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        return_node = heapq.heappop(self.queue)
        return (return_node[0],return_node[2])

    def remove(self, node_id):
        """
        Remove a node from the queue.

        Require this in ucs.

        Args:
            node_id (int): Index of node in queue.
        """

        for index, (c, _, n) in enumerate(self.queue):
            if node_id == n[-1]:
                temp = self.queue[index]
                del self.queue[index]
                return temp
                break
        heapq.heapify(self.queue)

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))


    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        val = node[0]
        heapq.heappush(self.queue, (val,self.size(),node[1]))
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    breadth-first-search.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start==goal:
        return []
    Explored_nodes = []
    frontier = [start]
    parent = {start:None}
    while frontier:
        if frontier==[]:
            return []
        Next = []
        for next_node in frontier:
            Explored_nodes.append(next_node)
            for child_node in graph.neighbors(next_node):
                if child_node not in Explored_nodes and child_node not in Next: 
                    if child_node not in frontier:
                        parent[child_node] = next_node
                        if child_node == goal:
                            solution = []
                            solution.append(child_node)
                            parent_state = parent[child_node]
                            while parent_state:
                                solution.append(parent_state)
                                parent_state = parent[parent_state] 
                            solution.reverse()
                            return solution
                        Next.append(child_node)
        frontier = sorted(Next)


def uniform_cost_search(graph, start, goal):
    """
    uniform_cost_search.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start==goal:
        return []
    Explored_nodes = []
    frontier = PriorityQueue()
    frontier.append((0,[start]))
    temp_frontier_nodes = []
    while frontier:
        cost,path = frontier.pop()
        next_node = path[-1]
        Explored_nodes.append(next_node)
        if next_node == goal:
            return path
        for child_node in graph.neighbors(next_node):
            new_path = None
            new_path = copy.deepcopy(path)
            new_path.append(child_node)
            total_cost = cost + graph.get_edge_weight(next_node, child_node)
            
            if child_node not in Explored_nodes: 
                if child_node in temp_frontier_nodes:
                    for c,_,p in frontier:
                        node = p[-1]
                        if child_node==node:
                            if total_cost<c:
                                frontier.remove(child_node)
                                frontier.append((total_cost,new_path))
                                break
                            else:
                                break
                else: 
                    frontier.append((total_cost,new_path))
                    temp_frontier_nodes.append(child_node)
    return []



def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    euclidean distance heuristic.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    if v == goal:
        return 0
    posv = graph.nodes[v]['pos']
    posgoal = graph.nodes[goal]['pos']
    #print(posv)
    #print(posgoal)
    distance = math.sqrt((posv[0]-posgoal[0])**2+(posv[1]-posgoal[1])**2)
    return distance
    
def get_total_cost(graph,path):
    cost = 0
    if path and len(path)>1:
        for i in range(len(path)-1):
            pre_node = path[i]
            post_node = path[i+1]
            cost += graph[pre_node][post_node]['weight']
    return cost

def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    A* algorithm.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start==goal or not graph.has_node(start) or not graph.has_node(goal):
        return []
    Explored_nodes = []
    frontier = PriorityQueue()
    frontier.append((0,[start]))
    while frontier:
        if frontier.size() == 0:
            return []
        cost,path = frontier.pop()
        next_node = path[-1]
        Explored_nodes.append(next_node)
        temp_frontier_nodes = []
        if next_node == goal:
            print(Explored_nodes)
            
            return path
        for child_node in graph.neighbors(next_node):
            new_path = None
            new_path = copy.deepcopy(path)
            new_path.append(child_node)
            total_cost = get_total_cost(graph,new_path)+heuristic(graph,child_node,goal)
            if child_node not in Explored_nodes: 
                if child_node in temp_frontier_nodes:
                    
                    
                    for c,_,p in frontier:
                        node = p[-1]
                        if child_node==node:
                            if total_cost<c:
                                frontier.remove(child_node)
                                frontier.append((total_cost,new_path))
                                break
                            else:
                                break
                else: 
                    
                    frontier.append((total_cost,new_path))
                    temp_frontier_nodes.append(child_node)
                print(temp_frontier_nodes)
            
    return []


def bidirectional_ucs(graph, start, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
