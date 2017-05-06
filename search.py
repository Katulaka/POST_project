import Queue as Q
import numpy as np

class Tag(object):
    def __init__(self, tag, score):
        self.tag = tag
        self.score = score


tags = np.empty((3,5),object)

for i in range(3):
    for j in range(5):
        tags[i][j] = Tag(str(i+j), i+j)



class State(object):
    
    def __init__(self, rid, lid, rank, tags):
        self.rid    = rid
        self.lid    = lid
        self.rank   = rank
        self.scr    = self.score(tags)
        self.hrstc  = self.heuristic(tags)
        self.total  = self.scr  + self.hrstc 
        return
    
    def is_valid(self, tags):
        return True #TODO

    def score(self, tags):
        idx_range   = range(self.rid, self.lid)
        pos         = zip(self.rank, idx_range)        
        return  0 if not self.is_valid(tags) else sum([tags[el].score for el in pos])

    def heuristic(self, tags):
        idx_range   = range(0,self.rid)+range(self.lid,tags.shape[1])
        rank        = [0]*len(idx_range)
        pos         = zip(rank, idx_range)
        return sum([tags[el].score for el in pos])

    def successor(self):


    def __cmp__(self, other):
        return -cmp(self.total, other.total)


class closed_list(object):

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


#TODO


def astar(tags, start, goal):
    
    frontier = Q.PriorityQueue()
    frontier.put(start)

#    came_from = {}
#    cost_so_far = {}
#    came_from[start] = None
 #   cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
#        for next in graph.neighbors(current):
#            new_cost = cost_so_far[current] + graph.cost(current, next)
#            if next not in cost_so_far or new_cost < cost_so_far[next]:
#                cost_so_far[next] = new_cost
#                priority = new_cost + heuristic(goal, next)
#                frontier.put(next, priority)
#                came_from[next] = current
#    
#    return came_from, cost_so_far


s1  = {'rid' : 0, 'lid' : 1, 'rank' : [1], 'tags' : tags}
s2  = {'rid' : 2, 'lid' : 4, 'rank' : [1,1,1], 'tags' : tags}
s3  = {'rid' : 0, 'lid' : 4, 'rank' : [1]*5, 'tags' : tags}

cl = closed_list()

cl.put(State(**s1))
cl.put(State(**s2))
cl.put(State(**s3))

print cl.get(2,4).rank


#A_star_search(tags, State(**start), goal):
#
#q.put(Skill(5, 'Proficient'))
#q.put(Skill(10, 'Expert'))
#q.put(Skill(1, 'Novice'))
#
#while not q.empty():
#    next_level = q.get()
#    print 'Processing level:', next_level.description
#



