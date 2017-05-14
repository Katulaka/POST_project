import Queue as Q
import numpy as np


class AStar:

    class Tag(object):
        def __init__(self, tag, score):
            self.tag    = tag
            self.score  = score

    class State(object):
    
        def __init__(self, data, gscore, fscore):
            self.data        = data
            self.gscore      = gscore
            self.fscore      = fscore
            self.closed      = False
            self.out_openset = True
            self.came_from   = None
    
        def __cmp__(self, other):
            return -cmp(self.fscore, other.fscore)


    class StateDict(dict):

        def __missing__(self, k):
            v = AStar.State(k)
            self.__setitem__(k, v)
            return v


    def is_valid(self, tags):
            return True #TODO

    def score(self, tree):
            idx_range   = range(tree[rid], tree[lid])
            pos         = zip(tree[rank], idx_range)        
            return  0 if not self.is_valid(tags) else sum([tags[el].score for el in pos])


    def is_goal_reached(self, current, goal):
        return (current[rid] == goal[rid] and current[lid] == goal[lid] and current[rank] == goal[rank])

    def heuristic(self, current, goal):
        idx_range   = range(0,current.data[rid])+range(current.data[lid], goal[lid])
        rank        = [0]*len(idx_range)
        pos         = zip(rank, idx_range)
        return sum([tags[el].score for el in pos])

    def successors(self, state):
        successors = []
        successors.append({rid:state[rid], lid:state[lid], rank:[state[rank]+1]})
        for idx in range(state[lid]+1, max_lid): #TODO
            for r in range(max_rank):
                successors.append({rid:state[rid], lid:idx, rank:state[rank].append(r)})

        return successors
    
    
    def astar(self, start, goal, tags):

        if self.is_goal_reached(start, goal):
            return [start]
        States = AStar.StateDict()
        startNode = States[start] = AStar.State(data, gscore=.0, fscore = self.heuristic(start, goal)) #TODO

        openSet = Q.PriorityQueue()
        openSet.put(start)

        while not openSet.empty():
            current = openSet.get()
            
            if self.is_goal_reached(current.data, goal):
                return #TODO
                 #self.reconstruct_path(current, reversePath)
#                break
            current.out_openset = True
            current.closed      = True
            for successor in [States[n] for n in self.successors(current.data)]:
                if successor.closed:
                    continue
                tentative_gscore = current.gscore + self.score(successor.data)
                if tentative_gscore <= successor.gscore:
                    continue
                successor.came_from = current
                successor.gscore = tentative_gscore
                successor.fscore = tentative_gscore + self.heuristic(successor.data, goal)
                if successor.out_openset:
                    successor.out_openset = False
                    openSet.put(successor)
#                    heappush(openSet, neighbor)
        return None
        
           

    class ClosedList(object):
    
        def __init__(self):
            self.rindex = {}
            self.lindex = {}
            return
    
        def put(self, State):
            if State.rid in  self.rindex:
                self.rindex[State.rid].append(State)
            else:
                self.rindex[State.rid] = [State]
    
            if State.lid in self.lindex:
                self.lindex[State.lid].append(State)
            else:
                self.lindex[State.lid] = [State]
            return
    
        #assumes unique States
        def get(self, rid, lid):
            if not rid in self.rindex or not lid in self.lindex:
                return None
            else:
                res = list(set(self.rindex[rid]).intersection(self.lindex[lid]))
                return  None if len(res)==0 else res[0]
    
    
    
        
tags = np.empty((3,5),object)
    
for i in range(3):
    for j in range(5):
        tags[i][j] = Tag(str(i+j), i+j)
    
    
    
s1  = {'rid' : 0, 'lid' : 1, 'rank' : [1], 'tags' : tags}
s2  = {'rid' : 2, 'lid' : 4, 'rank' : [1,1,1], 'tags' : tags}
s3  = {'rid' : 0, 'lid' : 4, 'rank' : [1]*5, 'tags' : tags}
    
cl = ClosedList()
    
cl.put(State(**s1))
cl.put(State(**s2))
cl.put(State(**s3))
    
print cl.get(2,4).rank
