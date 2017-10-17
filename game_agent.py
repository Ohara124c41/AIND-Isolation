
"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This should be the optimal heuristic function for your project submission.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #This heuristic is designed to evaluate the players distance relative to a
    #center move. It looks at the difference between the player and the opponent
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #Set up board and movements
    board = game.height * game.width
    moves_board = game.move_count / board
    move_player = len(game.get_legal_moves(player))
    move_opponent = len(game.get_legal_moves(game.get_opponent(player)))

    #Set up positions
    player_position = game.get_player_location(player)
    opponent_position = game.get_player_location(game.get_opponent(player))
    center_position = (int(game.height / 2), int(game.width / 2))

    #Set up distance formulas
    player_distance = abs(player_position[0] - center_position[0]) + abs(player_position[1] - center_position[1])
    opponent_distance = abs(opponent_position[0] - center_position[0]) + abs(opponent_position[1] - center_position[1])
    delta_center = opponent_distance - player_distance
    move_displacement = abs(player_position[0] - opponent_position[0]) + abs(player_position[1] - opponent_position[1])

    #Set up conditional statements
    if moves_board > 0.32:
        if move_player == move_opponent:
            move_difference = move_player - move_opponent*2 + delta_center
        else:
            move_difference = move_player - move_opponent*2
    else:
        if move_player == move_opponent:
            move_difference = move_player - move_opponent + delta_center
        else:
            move_difference = move_player - move_opponent

    #Return a value
    return float(move_difference / move_displacement)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """


    #This heuristic is designed evaluate movements based on empty/blank spaces
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #Set up board and movements
    move_player = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    blank_spaces = len(game.get_blank_spaces())
    board = game.height * game.width

    #Set up foreccasting
    for move in game.get_legal_moves(player):
        move_player += len(game.forecast_move(move).get_legal_moves(player))

    for move in game.get_legal_moves(game.get_opponent(player)):
        move_player += len(game.forecast_move(move).get_legal_moves(game.get_opponent(player)))

    #Return a value
    return float(move_player - ((1 + (blank_spaces/board)) * opponent_moves))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    #This heuristic is designed evaluate movements based on common sets 
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #Set up board and movements
    move_player = len(game.get_legal_moves(player))
    move_opponent = len(game.get_legal_moves(game.get_opponent(player)))
    board = game.height * game.width
    moves_board = game.move_count / board

    #Set up positions
    player_position = game.get_player_location(player)
    opponent_position = game.get_player_location(game.get_opponent(player))
    center_position = (int(game.height / 2), int(game.width / 2))

    #Set up distance formulas
    player_distance = abs(player_position[0] - center_position[0]) + abs(player_position[1] - center_position[1])
    opponent_distance = abs(opponent_position[0] - center_position[0]) + abs(opponent_position[1] - center_position[1])
    delta_center = opponent_distance - player_distance

    #Shared/Common set value, interested in the intersection
    commonality = len(set(player_position).intersection(set(opponent_position)))
    move_displacement = abs(player_position[0] - opponent_position[0]) + abs(player_position[1] - opponent_position[1])

    #Set up conditional statements
    if moves_board > 0.32:
        if move_player == move_opponent:
            move_difference = move_player - move_opponent*2 + delta_center + commonality
        else:
            move_difference = move_player - move_opponent*2
    else:
        if move_player == move_opponent:
            move_difference = move_player - move_opponent + delta_center + commonality
        else:
            move_difference = move_player - move_opponent

    #Return a value
    return float(move_difference / move_displacement)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout



class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the optimal move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        #Set up contingencies for illegal moves and timeouts
        optimal_move = (-1, -1)
        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass
        #Return the optimal move
        return optimal_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the optimal move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        _, move = max([(self.min_value(game.forecast_move(m),depth-1) , m) for m in legal_moves])
        return move

    def max_value(self,game,depth):
        legal_moves = game.get_legal_moves()
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if not legal_moves or depth == 0:
            return self.score(game,self)
        returned_value = max([ self.min_value(game.forecast_move(m),depth-1) for m in legal_moves])
        return returned_value

    def min_value(self,game,depth):
        legal_moves = game.get_legal_moves()
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if not legal_moves or depth == 0:
            return self.score(game,self)
        returned_value = min([ self.max_value(game.forecast_move(m),depth-1) for m in legal_moves])
        return returned_value

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the optimal move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        #Set up timer from AIMA documentation for each definition
        self.time_left = time_left

        #In the event of a timeout, use optimal move
        optimal_move = (-1, -1)
        iteration = True
        try:
            depth = 1
            while iteration:
                depth+=1
                optimal_move = self.alphabeta(game, depth)
        except SearchTimeout:
            pass

        # Return the optimal move
        return optimal_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the optimal move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        #Set up timer from AIMA documentation for each definition
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        optimal_move = (-1,-1)
        optimal_score = float('-inf')

        for m in legal_moves:
            returned_value = self.min_value(game.forecast_move(m),depth-1,alpha,beta)
            if returned_value > optimal_score:
                optimal_move = m
                optimal_score = returned_value
            alpha = max(alpha,returned_value)
        return optimal_move

    def min_value(self,game,depth,alpha,beta):
        legal_moves = game.get_legal_moves()
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if not legal_moves or depth == 0:
            return self.score(game,self)
        returned_value = float('inf')
        for m in legal_moves:
            returned_value = min(returned_value,self.max_value(game.forecast_move(m),depth-1,alpha,beta))
            if returned_value <= alpha:
                return returned_value
            beta = min(beta,returned_value)
        return returned_value

    def max_value(self,game,depth,alpha,beta):
        legal_moves = game.get_legal_moves()
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if not legal_moves or depth == 0:
            return self.score(game,self)
        returned_value = float('-inf')
        for m in legal_moves:
            returned_value = max(returned_value,self.min_value(game.forecast_move(m),depth-1,alpha,beta))
            if returned_value >= beta:
                return returned_value
            alpha = max(alpha,returned_value)
        return returned_value
