from board import Board
from search import SearchProblem
import util
import numpy as np


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, starting_point=self.starting_point)
        self.targets = [(0, 0), (0, board_w - 1), (board_h - 1, 0), (board_h - 1, board_w - 1)]

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        board = state.state
        for target in self.targets:
            if board[target] == -1:
                return False
        return True
        # return np.all(board[[0, 0, -1, -1], [0, -1, 0, -1]] != -1)

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        total_cost = 0
        for action in actions:
            cost = action.piece.get_num_tiles()
            total_cost += cost
        return total_cost


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def linear_distance(point1, point2):
    return np.max(np.abs(np.array(point1) - np.array(point2))) + 1


def get_valid_adjacent_tiles(tile, board):
    tiles = [np.array(tile)]
    board_h, board_w = board.shape
    if tile[0] + 1 < board_h:
        tiles.append(np.array([tile[0] + 1, tile[1]]))
    if tile[0] - 1 >= 0:
        tiles.append(np.array([tile[0] - 1, tile[1]]))
    if tile[1] + 1 < board_w:
        tiles.append(np.array([tile[0], tile[1] + 1]))
    if tile[1] - 1 >= 0:
        tiles.append(np.array([tile[0], tile[1] - 1]))

    return tiles


def fill_tiles_in_the_way(tile1, tile2, board):
    tile1 = np.array(tile1).astype(np.int)
    tile2 = np.array(tile2).astype(np.int)

    while True:
        board[tile1[0], tile1[1]] = 0

        vec = tile1 - tile2
        curr_interpolation_vec = np.array([0, 0])
        if vec[0] < 0:
            curr_interpolation_vec[0] = 1
        elif vec[0] > 0:
            curr_interpolation_vec[0] = -1

        if vec[1] < 0:
            curr_interpolation_vec[1] = 1
        elif vec[1] > 0:
            curr_interpolation_vec[1] = -1

        tile1 = tile1 + curr_interpolation_vec

        if np.abs(vec).sum() == 0:
            return


def get_goal_tiles(state, problem):
    goals = set()
    for target in problem.targets:
        if state.state.item(target) == -1:
            goals.add(target)

    return goals


def get_legal_tiles(state, problem):
    legal_tiles = set()

    if np.all(state.state == -1):
        legal_tiles.add(problem.starting_point)
    else:
        for y in range(state.board_h):
            for x in range(state.board_w):
                if state.state.item((x, y)) == -1 and state.check_tile_attached(0, x, y) and state.check_tile_legal(0,
                                                                                                                    x,
                                                                                                                    y):
                    legal_tiles.add((y, x))

    return legal_tiles


def is_tile_illegal(tile, state):
    tiles = [np.array(tile) for _ in range(4)]
    interpolation_vecs = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]
    check_tiles = []

    for i in range(4):
        tiles[i] += interpolation_vecs[i]
        if not (np.any(tiles[i] < 0) or tiles[i][0] > state.board_w - 1 or tiles[i][1] > state.board_h - 1):
            check_tiles.append(tiles[i])

    for check_tile in check_tiles:
        if state.state[check_tile[0], check_tile[1]] != -1:
            return True

    return False


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    goals = get_goal_tiles(state, problem)
    legal_tiles = get_legal_tiles(state, problem)

    if len(goals) == 0:
        return 0

    if len(legal_tiles) == 0:
        return np.inf

    # Closest tiles to goal, amount of tiles to change in order to get to solution
    pairs = dict()
    dists = dict()
    for goal in goals:
        if is_tile_illegal(goal, state):
            return np.inf
        for legal_tile in legal_tiles:
            dist = euclidean_distance(goal, legal_tile)
            if (goal in pairs.keys() and dists[goal] > dist) or goal not in pairs.keys():
                pairs[goal] = legal_tile
                dists[goal] = dist

    max_dist = 0
    for goal, tile in pairs.items():
        board = np.copy(state.state)
        fill_tiles_in_the_way(goal, tile, board)
        min_dist = (board != state.state).sum()
        max_dist = max(min_dist, max_dist)

    # min_dist = np.inf
    #
    # for goal in goals:
    #     all_dist = 0
    #     for legal_tile in legal_tiles:
    #         board = np.copy(state.state)
    #         fill_tiles_in_the_way(goal, legal_tile, board)
    #         all_dist = max(all_dist, (board != state.state).sum())
    #     min_dist = min(all_dist, min_dist)
    #
    # result = max(max_dist, min_dist)
    # print(result)
    return min(max_dist, problem.board_h + problem.board_w - 1)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=None):
        if targets is None:
            targets = [(0, 0)]
        self.targets = targets.copy()
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, starting_point=self.starting_point)
        distances = []
        for target_1 in targets:
            for target_2 in targets:
                distances.append(linear_distance(target_1, target_2))
        self.max_distance = max(distances)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        done = True
        for target in self.targets:
            done &= (state.state.item(target) != -1)
        return done

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        total_cost = 0
        for action in actions:
            cost = action.piece.get_num_tiles()
            total_cost += cost
        return total_cost


def blokus_cover_heuristic(state, problem):
    goals = get_goal_tiles(state, problem)
    legal_corners = get_legal_tiles(state, problem)

    pairs = dict()
    dists = dict()
    for goal in goals:
        for legal_corner in legal_corners:
            dist = euclidean_distance(goal, legal_corner)
            if (goal in pairs.keys() and dists[goal] > dist) or goal not in pairs.keys():
                pairs[goal] = legal_corner
                dists[goal] = dist

    max_dist = 0
    for goal, corner in pairs.items():
        board = np.copy(state.state)
        fill_tiles_in_the_way(goal, corner, board)
        min_dist = (board != state.state).sum()
        max_dist = max(min_dist, max_dist)

    return min(max_dist, problem.max_distance)


from search import GraphNode


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, starting_point=self.starting_point)
        distances = []
        for target_1 in targets:
            for target_2 in targets:
                distances.append(linear_distance(target_1, target_2))
        self.max_distance = max(distances)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def heuristic(self, state, curr_goal):
        legal_tiles = get_legal_tiles(state, self)

        if state.state.item(curr_goal) != -1:
            return -np.inf

        if not state.check_tile_legal(0, curr_goal[0], curr_goal[1]):
            return np.inf

        dists = []
        for tile in legal_tiles:
            dists.append(linear_distance(curr_goal, tile))

        return min(dists)

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        backtrace = []
        starting_states = []
        start_state = self.get_start_state()
        start_node = GraphNode(None, start_state, None, 0, self.heuristic(start_state, self.targets[0]))
        starting_states.append(start_node)
        visited_list = set()
        i = 0

        while i < len(self.targets):
            fringe = util.PriorityQueueWithFunction(lambda x: x.heuristic_cost)
            fringe.push(start_node)
            stop = False

            while fringe:
                node = fringe.pop()
                if node.state in visited_list:
                    continue
                else:
                    visited_list.add(node.state)

                if not node.state.check_tile_legal(0, self.targets[i][0], self.targets[i][1]):
                    break
                legal_actions = self.get_successors(node.state)
                for triplet in legal_actions:
                    if triplet[0] in visited_list:
                        continue
                    h = self.heuristic(triplet[0], self.targets[i])
                    curr_node = GraphNode(node, triplet[0], triplet[1], triplet[2],
                                          heuristic_cost=h)
                    fringe.push(curr_node)
                    if h == -np.inf:
                        backtrace.append(curr_node.get_moves())
                        start_node = curr_node
                        starting_states.append(start_node)
                        stop = True
                        break
                if stop:
                    break

            i += 1
            if not stop:
                # No more moves, revert back one step
                del backtrace[-1]
                del starting_states[-1]
                start_node = starting_states[-1]
                visited_list.remove(start_node.state)
                i -= 2

        return backtrace[-1]


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, starting_point=self.starting_point)
        distances = []
        for target_1 in targets:
            for target_2 in targets:
                distances.append(linear_distance(target_1, target_2))
        self.max_distance = max(distances)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def heuristic(self, state, curr_goal):
        legal_tiles = get_legal_tiles(state, self)

        if state.state.item(curr_goal) != -1:
            return -np.inf

        if not state.check_tile_legal(0, curr_goal[0], curr_goal[1]):
            return np.inf

        dists = []
        for tile in legal_tiles:
            dists.append(linear_distance(curr_goal, tile))

        return min(dists)

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        backtrace = []
        starting_states = []
        start_state = self.get_start_state()
        start_node = GraphNode(None, start_state, None, 0, self.heuristic(start_state, self.targets[0]))
        starting_states.append(start_node)
        visited_list = set()
        i = 0

        while i < len(self.targets):
            fringe = util.PriorityQueueWithFunction(lambda x: x.heuristic_cost)
            fringe.push(start_node)
            stop = False

            while fringe:
                node = fringe.pop()
                if node.state in visited_list:
                    continue
                else:
                    visited_list.add(node.state)

                if not node.state.check_tile_legal(0, self.targets[i][0], self.targets[i][1]):
                    break
                legal_actions = self.get_successors(node.state)
                for triplet in legal_actions:
                    if triplet[0] in visited_list:
                        continue
                    h = self.heuristic(triplet[0], self.targets[i])
                    curr_node = GraphNode(node, triplet[0], triplet[1], triplet[2],
                                          heuristic_cost=h)
                    fringe.push(curr_node)
                    if h == -np.inf:
                        backtrace.append(curr_node.get_moves())
                        start_node = curr_node
                        starting_states.append(start_node)
                        stop = True
                        break
                if stop:
                    break

            i += 1
            if not stop:
                # No more moves, revert back one step
                del backtrace[-1]
                del starting_states[-1]
                start_node = starting_states[-1]
                visited_list.remove(start_node.state)
                i -= 2

        return backtrace[-1]
