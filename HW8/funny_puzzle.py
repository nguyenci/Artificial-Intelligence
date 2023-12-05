import heapq
import numpy as np
import copy

"""
HW8
Author: Cinthya Nguyen
Class: CS540 SP23
"""

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
        INPUT:
            Two states (if second state is omitted then it is assumed that it is the goal state)

        RETURNS:
            A scalar that is the sum of Manhattan distances for all tiles.
        """
    from_state = np.reshape(from_state, (3, 3))
    to_state = np.reshape(to_state, (3, 3))

    distance = 0
    for i in range(0, 3):
        for j in range(0, 3):
            item = from_state[i][j]

            if item == 0 or to_state[i][j] == from_state[i][j]:
                continue

            arr = np.where(to_state == item)
            x = arr[0][0]
            y = arr[1][0]
            distance += (abs(x - i) + abs(y - j))

    return distance


def print_succ(state):
    """
     INPUT:
         A state (list of length 9)

     WHAT IT DOES:
         Prints the list of all the valid successors in the puzzle.
     """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    INPUT:
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done
        below).
    """
    state = np.reshape(state, (3, 3))
    arr = np.where(state == 0)

    x1 = arr[0][0]
    y1 = arr[1][0]
    x2 = arr[0][1]
    y2 = arr[1][1]

    # Get the adjacent tiles
    s_1 = find_adj(x1, y1, state)  # State 1
    s_2 = find_adj(x2, y2, state)  # State 2

    succ_states = []
    for i in s_1:
        temp = copy.deepcopy(state)
        temp[x1][y1] = state[i[0]][i[1]]
        temp[i[0]][i[1]] = 0
        temp = temp.reshape(1, 9)
        succ_states.append(list(temp[0]))

    for i in s_2:
        temp = copy.deepcopy(state)
        temp[x2][y2] = state[i[0]][i[1]]
        temp[i[0]][i[1]] = 0
        temp = temp.reshape(1, 9)
        succ_states.append(list(temp[0]))

    return sorted(succ_states)


def find_adj(x, y, state):
    """
    Helper function to get the adjacent tiles of a given tile.
    :param x: x coordinate of the tile
    :param y: y coordinate of the tile
    :param state: current state
    :return: adjacent tiles
    """
    result = []
    if valid(x - 1, y):
        if state[x - 1][y] != 0:
            result.append(((x - 1), y))
    if valid(x, y - 1):
        if state[x][y - 1] != 0:
            result.append((x, (y - 1)))
    if valid(x, y + 1):
        if state[x][y + 1] != 0:
            result.append((x, (y + 1)))
    if valid(x + 1, y):
        if state[x + 1][y] != 0:
            result.append(((x + 1), y))

    return result


def valid(x, y):
    """
    Helper function to check if a tile is valid or not.
    :param x: x coordinate of the tile
    :param y: y coordinate of the tile
    :return: 1 if valid, 0 otherwise
    """
    if (x < 0) or (y < 0) or (x > 2) or (y > 2):
        return 0
    return 1


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
        INPUT:
            An initial state (list of length 9)

        WHAT IT SHOULD DO:
            Prints a path of configurations from initial state to goal state along  h values,
            number of moves, and max queue number in the format specified in the pdf.
        """
    queue = []
    visited = []

    parent = -1
    final = dict()
    g = 0
    h = get_manhattan_distance(state, goal_state)
    cost = g + h
    pid = 0
    heapq.heappush(queue, (cost, state, (g, h, pid, parent)))

    while len(queue) > 0:
        curr = heapq.heappop(queue)  # Get first item in queue
        visited.append(curr[1])  # Add to visited list
        final[curr[2][2]] = []
        final[curr[2][2]].append((curr[1], curr[2][1], curr[2][0], curr[2][3]))

        if curr[1] == goal_state:  # Check if goal state is reached
            break

        parent = curr[2][2]
        succ_states = get_succ(curr[1])

        for i in succ_states:
            if i in visited:
                continue
            else:
                g = curr[2][0] + 1
                h = get_manhattan_distance(i, goal_state)
                cost = g + h
                pid += 1
                heapq.heappush(queue, (cost, i, (g, h, pid, parent)))

    path = []
    last_key = list(final.keys())[-1]

    while last_key != -1:
        path.append(final[last_key])
        parent = final[last_key][0][3]
        last_key = parent

    path_reversed = list(reversed(path))

    for i in path_reversed:
        print(f'{i[0][0]} h={i[0][1]} moves: {i[0][2]}')

    print("Max queue length:", len(queue) + 1)


if __name__ == "__main__":
    pass
