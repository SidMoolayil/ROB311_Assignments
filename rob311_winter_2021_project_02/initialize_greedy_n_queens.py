import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    # initialization
    greedy_init = np.zeros(N)

    # one queen per column, .'. only horizontal and diagonal fills need to be considered
    diag1Fill = np.zeros([2*N-1])
    diag2Fill = np.zeros([2*N-1])
    horFill = np.zeros(N)

    for i in range (0,N):

        # sum fill vectors for new min position (diagonals and horizontal queen conflicts weighted same)
        tempColRes = horFill + diag1Fill[i:N+i] + diag2Fill[(N-1-i):(2*N-i-1)]

        # candidate rows all of which are minimum conflicts (greedy criterion) with other queens
        candidate = np.flatnonzero(tempColRes == tempColRes.min())

        # randomly choose among the candidate rows
        newRow = np.random.choice(candidate)
        greedy_init[i] = newRow

        # register new position as filled (horizontal and 2 diagonals)
        diag1Fill[i+newRow] += 1
        diag2Fill[newRow- i + N-1] += 1
        horFill[newRow] += 1

    # ensure data is of type int
    greedy_init = np.ndarray.astype(greedy_init, int)
    ### YOUR CODE GOES HERE
    return greedy_init


if __name__ == '__main__':
    np.random.seed(0)
    q=initialize_greedy_n_queens(10)
    print(q, "final")

    # You can test your code here
    pass
