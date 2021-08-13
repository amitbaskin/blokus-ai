"""
In search.py, you will implement generic search algorithms
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


class GraphNode:
    originator_node = None
    state = None
    move = None
    cost = 0
    heuristic_cost = 1
    cost_so_far = 0
    astar_cost = 0

    def __init__(self, origin_node, state, move, cost, heuristic_cost=1):
        self.originator_node = origin_node
        self.state = state
        self.move = move
        self.cost = cost
        self.heuristic_cost = heuristic_cost
        self.cost_so_far = self.originator_node.cost_so_far + cost if self.originator_node is not None else cost
        self.astar_cost = heuristic_cost + self.cost_so_far

    def get_moves(self):
        # Get steps by going up the path to origin
        steps = []
        curr_node = self
        while curr_node.originator_node is not None:
            steps.append(curr_node.move)
            curr_node = curr_node.originator_node
        steps.reverse()
        return steps

    def __eq__(self, other):
        return self.astar_cost == other.astar_cost

    def __gt__(self, other):
        return self.astar_cost > other.astar_cost

    def __lt__(self, other):
        return self.astar_cost < other.astar_cost


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def graph_search_pattern(fringe, problem, insertion_func, heuristic=null_heuristic):
    visited_list = set()
    start_state = problem.get_start_state()
    start_node = GraphNode(None, start_state, None, 0, heuristic(start_state, problem))
    fringe.push(start_node)

    while fringe:
        node = fringe.pop()
        if node.state in visited_list:
            continue
        else:
            visited_list.add(node.state)
        if problem.is_goal_state(node.state):
            return node.get_moves()
        else:
            legal_actions = problem.get_successors(node.state)
            insertion_func(legal_actions, fringe, node)

    return []


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """

    def dfs_insertion_func(legal_action_triplets, curr_fringe, curr_node):
        for triplet in legal_action_triplets:
            curr_fringe.push(GraphNode(curr_node, triplet[0], triplet[1], triplet[2]))

    fringe = util.Stack()
    return graph_search_pattern(fringe, problem, dfs_insertion_func)


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    def bfs_insertion_func(legal_action_triplets, curr_fringe, curr_node):
        for triplet in legal_action_triplets:
            curr_fringe.push(GraphNode(curr_node, triplet[0], triplet[1], triplet[2]))

    fringe = util.Queue()
    return graph_search_pattern(fringe, problem, bfs_insertion_func)


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    def uniform_insertion_func(legal_action_triplets, curr_fringe, curr_node):
        for triplet in legal_action_triplets:
            curr_fringe.push(GraphNode(curr_node, triplet[0], triplet[1], triplet[2]))

    fringe_queue = util.PriorityQueueWithFunction(lambda x: x.cost_so_far)
    return graph_search_pattern(fringe_queue, problem, uniform_insertion_func)


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    def astar_insertion_func(legal_action_triplets, curr_fringe, curr_node):
        for triplet in legal_action_triplets:
            node = GraphNode(curr_node, triplet[0], triplet[1], triplet[2],
                             heuristic_cost=heuristic(triplet[0], problem))
            curr_fringe.push(node)

    fringe_queue = util.PriorityQueueWithFunction(lambda x: x.astar_cost)
    return graph_search_pattern(fringe_queue, problem, astar_insertion_func, heuristic)


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
