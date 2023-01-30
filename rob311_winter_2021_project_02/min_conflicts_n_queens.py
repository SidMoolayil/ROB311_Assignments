import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

def conflict(queen, square):
    # print (square,np.sum(square),queen, np.sum(queen))
    if queen[0]==square[0] or queen[1]==square[1] or np.sum(square)==np.sum(queen) or queen[0] - square[0] == queen[1] - square[1]:
        return True
    # print(queen[0] == square[0], queen[1] == square[1], np.sum(square) == np.sum(queen),
    #       queen[0] - square[0] == queen[1] - square[1])

    return False

def anyConflict(queens, square):
    for i in range(len(queens)):
        # print("queen#",i, square)
        if square[0]==i :
            # print(square[0]==queens[i] and square[1]==i)
            continue
        if conflict((i,queens[i]),square):

            return (i,queens[i])

    return False

def colConflict2(board, col):
    # print("in colconflict", board, col)
    N=board.shape[0]
    hori=np.sum(board,axis=1)
    # vert=np.sum(board[:,col])
    flipped=np.flip(board, axis=1)
    diag=np.array([np.trace(board,offset=i)+np.trace(flipped, offset=i-2*col-1+N) for i in range(col,-N+col,-1)])
    # diag1 = np.array([np.trace(board, offset=i)for i in range(col,-N+col,-1)])
    # diag2=np.array([np.trace(flipped, offset=i-2*col-1+N) for i in range(col,-N+col,-1)])
    # print(diag1,diag2)
    # print(hori,diag)
    return hori+diag

def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """
    # number of queens
    N = len(initialization)
    # initialization of solution
    solution = initialization.copy()

    max_steps = 1000

    # initialization
    queen = np.zeros([N, 3]) # i'th queen position defined by the horizontal and diagonals its occupying
    diag1Fill = np.zeros([2 * N - 1])
    diag2Fill = np.zeros([2 * N - 1])
    horFill = np.zeros(N)

    for i in range(N):
        queen[i] = [initialization[i], i + initialization[i], initialization[i] - i + N - 1] # i'th queen hor, diag1, diag 2 positions
        # count the hor, diag1, and diag 2 filled by initialized queens
        diag1Fill[int(queen[i, 1])] += 1
        diag2Fill[int(queen[i, 2])] += 1
        horFill[int(queen[i, 0])] += 1

    for idx in range(max_steps):
        done = True
        # loop over N queens
        for i in range(N):
            # if no conflicts, do not move (i.e. skip) the i'th queen
            if diag1Fill[int(queen[i, 1])] == 1 and diag2Fill[int(queen[i, 2])] == 1 and horFill[int(queen[i, 0])] == 1:
                continue

            # sum the fill vectors for new min position (diagonals and horizontal queen conflicts weighted same)
            tempColRes = horFill + diag1Fill[i:N + i] + diag2Fill[N - 1 - i:2 * N - i - 1]

            # if a conflict exists, process is not finished yet
            if tempColRes.max() > 3:
                done = False

            # candidate rows all of which are minimum conflicts (greedy criterion) with other queens
            candidate = np.flatnonzero(tempColRes == tempColRes.min())

            # randomly choose among the candidate rows
            solution[i] = np.random.choice(candidate)

            # remove queen from old position
            diag1Fill[int(queen[i, 1])] -= 1
            diag2Fill[int(queen[i, 2])] -= 1
            horFill[int(queen[i, 0])] -= 1

            # add queen to new position
            queen[i] = [solution[i], i + solution[i], solution[i] - i + N - 1]
            diag1Fill[int(queen[i, 1])] += 1
            diag2Fill[int(queen[i, 2])] += 1
            horFill[int(queen[i, 0])] += 1

        # if no conflicts exist
        if done:
            num_steps = idx
            print(solution, num_steps)
            return solution, num_steps

    return [],-1



if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    np.random.seed(1)
    N = 7
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    print(assignment_initial)
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)
