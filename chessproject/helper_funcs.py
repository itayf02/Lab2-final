import chess
def is_terminal(board: chess.Board) -> bool:
    """
    Check if the given board state is a terminal state.

    Parameters:
    board (chess.Board): The current state of the chess game.

    Returns:
    bool: True if the game is in a terminal state, False otherwise.
    """
    # 1. Checkmate
    if board.is_checkmate():
        return True

    # 2. Stalemate
    if board.is_stalemate():
        return True

    # 3. Insufficient material
    if board.is_insufficient_material():
        return True

    # 4. Threefold repetition
    if board.is_repetition(3):
        return True

    # 5. Fifty-move rule
    if board.is_fifty_moves():
        return True

    # If none of the terminal conditions are met, it's not a terminal state
    return False

def game_res(board):
    if board.outcome().result() =='1-0':
        return 1
    elif board.outcome().result() =='0-1':
        return -1
    else:
        return 0

def evaluate(board):
    """
    Evaluates the board position based on material advantage.

    Args:
        board: A chess.Board object representing the current board state.

    Returns:
        An integer representing the material advantage of the side to move.
        Positive values indicate an advantage for white, negative for black.
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King's value is not considered in material evaluation
    }

    material_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                material_score += piece_values.get(piece.piece_type, 0)
            else:
                material_score -= piece_values.get(piece.piece_type, 0)

    return material_score

def get_children(board):
    children = []
    for move in board.legal_moves:
        child_board = board.copy()
        child_board.push(move)
        children.append(child_board)
    return children