from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score


def nashProb(game_matrix):
    size = game_matrix.shape[0]
    matrix = np.copy(game_matrix)
    rowDiff = np.zeros((size - 1, size))
    np.fill_diagonal(rowDiff, 1)
    np.fill_diagonal(rowDiff[:, 1:], -1)

    matrix = rowDiff @ matrix
    matrix = np.vstack((matrix, np.ones((1, size))))
    b = np.zeros((size, 1))
    b[-1, -1] = 1
    # print(matrix,b)
    return np.squeeze(np.linalg.solve(matrix, b))


def seePattern(mode, previous, threshold, gameMat):
    c = 0
    if mode == 2:
        c = np.sum([previous[i][1] == np.argmin(gameMat[previous[i - 1][0], :]) for i in range(1, len(previous))])
    else:
        c = np.sum([previous[i][1] == previous[i - 1][mode] for i in range(1, len(previous))])

    if c > threshold - 2:
        print(mode, "pattern")
        return True
    return False


class StudentAgent(IteratedGamePlayer):
    """
    YOUR DOCUMENTATION GOES HERE!
    play randomly for some time to see any patterns in opponent (goldfish, copy, firstmove)
    (goldfish info was obtained through autolab testing)

    otherwise, there is no easy counter so use general method.
    See functions above (nashProb and seePattern)

    Use nash equilibrium initialization (self.gameChoice)
    add in a memory initialized to 1 (self.memChoice)
    the probability distribution (eduChoice) is obtained multiply gameChoice and memChoice, and then normalizing
    np.random.randomsample is used to pick a move
    the self.memChoice's value of the move is updated by multiplying by 2^(result/average(abs(gamematrix))).
    exponential because it will never lead to zero and works well with probabilities.
    /average(abs(gamematrix)) to make sure it's not too sudden of a change.
    """

    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        # YOUR CODE GOES HERE
        # print(game_matrix,"=gamematrix")
        self.gameChoice = nashProb(game_matrix)
        # self.gameChoice=np.squeeze(np.sum(game_matrix, axis=1))
        # self.gameChoice=(self.gameChoice+np.abs(np.min(self.gameChoice))+1)
        # self.gameChoice=self.gameChoice/np.sum(self.gameChoice)

        self.avResult = np.mean(np.abs(self.game_matrix))
        self.threshold = 5

        self.copyDetect = False
        self.firstMoveDetect = False
        self.goldfishDetect = False
        self.previous = []
        self.memChoice = [1 for i in range(self.n_moves)]

        # print(self.avResult)
        pass

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        # YOUR CODE GOES HERE
        if len(self.previous) < self.threshold:
            return np.random.randint(0, self.n_moves - 1)
        if len(self.previous) == self.threshold:
            print("confirming copy/firstmove")
            print(self.previous)
            self.copyDetect = seePattern(0, self.previous, self.threshold, self.game_matrix)
            self.firstMoveDetect = seePattern(1, self.previous, self.threshold, self.game_matrix)
            self.goldfishDetect = seePattern(2, self.previous, self.threshold, self.game_matrix)

        if self.copyDetect:
            return np.argmax(self.game_matrix[:, self.previous[-1][0]])
        if self.firstMoveDetect:
            return np.argmax(self.game_matrix[:, self.previous[-1][1]])
        if self.goldfishDetect:
            return np.argmax(self.game_matrix[:, np.argmin(self.game_matrix[self.previous[-1][0]])])

        r = np.random.random_sample()
        # print("r=",r)
        eduChoice = (self.gameChoice * self.memChoice)
        # eduChoice=(self.gameChoice)
        eduChoice = eduChoice / np.sum(eduChoice)
        # print("edu", eduChoice)
        eduChoice = np.cumsum(eduChoice)
        # print(self.gameChoice, self.memChoice,eduChoice)

        # print(eduChoice)
        # check which move to make
        for i in range(len(self.gameChoice) - 1):
            # print(type(r),type(r))
            if eduChoice[i] > r:
                # print("pick", i)
                return i

        # print("pick", len(self.gameChoice)-1)
        return len(self.gameChoice) - 1
        pass

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # YOUR CODE GOES HERE
        # heuristic
        # print(self.game_matrix[my_move, other_move], my_move,other_move)
        self.previous.append([my_move, other_move])

        if len(self.previous) > self.threshold: self.memChoice[my_move] = self.memChoice[my_move] * np.exp2(self.game_matrix[my_move, other_move] / self.avResult)
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.copyDetect = False
        self.previous = []
        self.firstMoveDetect = False
        self.memChoice = [1 for i in range(self.n_moves)]
        self.goldfishDetect = False
        np.random.seed(0)
        pass


if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    np.random.seed(0)
    a=3.0
    b=1.3
    c=1.7
    game_matrix = np.array([[0.0, -a, b],
                            [a, 0.0, -c],
                            [-b, c, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    copycat_player= CopycatPlayer (game_matrix)
    player=uniform_player
    # uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix,5)

    # print("Uniform player's score: {:}".format(uniform_score))
    # print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score, score = play_game(student_player, player, game_matrix,100)

    print("Your player's score: {:}".format(student_score))
    print("{} player's score: {}".format(player, score))
