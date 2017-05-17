# -*- coding: utf-8 -*-
""" generic A-Star path searching algorithm """

from abc import ABCMeta, abstractmethod
from heapq import heappush, heappop


class AStar:
    __metaclass__ = ABCMeta
    __slots__ = ()

    class SearchNode:
        __slots__ = ('data', 'fscore',
                     'closed', 'came_from', 'out_openset')

        def __init__(self, data, fscore=.0):
            self.data = data
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, other):
            return self.fscore > other.fscore


    class SearchNodeDict(dict):

        def __missing__(self, k):
            v = AStar.SearchNode(k)
            self.__setitem__(k, v)
            return v

    @abstractmethod
    def heuristic_cost(self, current, goal):
        """Computes the estimated (rough) distance between a node and the goal, this method must be
        implemented in a subclass. The second parameter is always the goal."""
        raise NotImplementedError

    @abstractmethod    
    def real_cost(self, current):
        raise NotImplementedError        

    @abstractmethod
    def neighbors(self, node):
        """For a given node, returns (or yields) the list of its neighbors. this method must be
        implemented in a subclass"""
        raise NotImplementedError

    @abstractmethod
    def is_goal_reached(self, current, goal):
        """ returns true when we can consider that 'current' is the goal"""
        raise NotImplementedError

    @abstractmethod
    def move_to_closed(self, current):
        raise NotImplementedError
        
    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.data
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def astar(self, start, goal, reversePath=False):
        searchNodes = AStar.SearchNodeDict()
        openSet = []
        for strt in  start:
            if self.is_goal_reached(strt, goal):
                return [strt]
            startNode = searchNodes[strt] = AStar.SearchNode(
               strt, fscore=self.real_cost(strt) + self.heuristic_cost(strt, goal))
            heappush(openSet, startNode)
        while openSet:
            current = heappop(openSet)
            print "---------------------------------------------------"
            print "current:", current.data.idx, current.data.rank, current.fscore
            if self.is_goal_reached(current.data, goal):
                return self.reconstruct_path(current, reversePath)
            current.out_openset = True
            current.closed = True
            self.move_to_closed(current.data)
            for neighbor in [searchNodes[n] for n in self.neighbors(current.data)]:
                
                if neighbor.closed:
                    continue
               
                neighbor.fscore = self.real_cost(neighbor.data) + \
                        self.heuristic_cost(neighbor.data, goal)
                neighbor.came_from = current
                
              #  if current.fscore > neighbor.fscore:
              #      continue

                if neighbor.out_openset:
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                print "neighbor:",neighbor.data.idx, neighbor.data.rank, neighbor.fscore
        return None


# def find_path(start, goal, neighbors_fnct, reversePath=False, heuristic_cost_estimate_fnct=lambda a, b: Infinite,
# distance_between_fnct=lambda a, b: 1.0, is_goal_reached_fnct=lambda a, b: a == b):
#    """A non-class version of the path finding algorithm"""
#    class FindPath(AStar):
#        def heuristic_cost_estimate(self, current, goal):
#            return heuristic_cost_estimate_fnct(current, goal)
#        def distance_between(self, n1, n2):
#            return distance_between_fnct(n1, n2)
#        def neighbors(self, node):
#            return neighbors_fnct(node)
#        def is_goal_reached(self, current, goal):
#            return is_goal_reached_fnct(current, goal)
#    return FindPath().astar(start, goal, reversePath)
#
# __all__ = ['AStar', 'find_path']
