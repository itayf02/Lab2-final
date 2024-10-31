# alpha_beta_pruning.pyx
import math
import chess  # Standard Python import for chess
from libc.math cimport INFINITY as INF
from cython cimport bint, double

# Alpha-beta pruning function with generic object type for board
cpdef double alpha_beta_pruning(object board, int depth, double alpha, double beta, bint maximizing_player):
    if depth == 0 or is_terminal(board):
        if is_terminal(board):
            return game_res(board) * INF
        return evaluate(board)

    cdef double eval, max_eval, min_eval
    if maximizing_player:
        max_eval = -INF
        for child in get_children(board):
            eval = alpha_beta_pruning(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = INF
        for child in get_children(board):
            eval = alpha_beta_pruning(child, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval

# Optimized helper functions
cdef bint is_terminal(object board):
    # Combine checks in a single statement for efficiency
    return board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3) or board.is_fifty_moves()

cdef double game_res(object board):
    outcome = board.outcome()
    if outcome is not None:
        result = outcome.result()
        if result == '1-0':
            return 1.0
        elif result == '0-1':
            return -1.0
    return 0.0  # Draw or no outcome

cdef double evaluate(object board):
    cdef dict piece_values = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0}
    cdef double material_score = 0.0
    cdef int square

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                material_score += piece_values.get(piece.piece_type, 0.0)
            else:
                material_score -= piece_values.get(piece.piece_type, 0.0)

    return material_score

cdef list get_children(object board):
    cdef list children = []
    for move in board.legal_moves:
        child_board = board.copy()
        child_board.push(move)
        children.append(child_board)
    return children
