import pandas as pd
import numpy as np
import torch
import torch_geometric
import torch_geometric.utils as pyg_utils
from torch.utils.data import Dataset
import chess
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,GINConv,TransformerConv,global_add_pool, global_mean_pool,global_max_pool,max_pool_neighbor_x
from torch_geometric.loader import DataListLoader,DataLoader
from sklearn.metrics import recall_score
# from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from info_nce import InfoNCE, info_nce
from pytorch_metric_learning import losses
import warnings
import chess.pgn
import chess
from tqdm import tqdm
import io
import random
import math

def chess_position_to_graph(board):
    G = nx.DiGraph()  # Directed graph

    # Ensure all squares are included as nodes
    for square in chess.SQUARES:
        G.add_node(chess.square_name(square))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get all squares attacked by the piece on the current square
            attacked_squares = board.attacks(square)
            for target_square in attacked_squares:
                # Add a directed edge from the attacking square to the attacked square
                from_square_name = chess.square_name(square)
                to_square_name = chess.square_name(target_square)
                G.add_edge(from_square_name, to_square_name)
    # Get the legal moves for the current player
    current_moves = list(board.legal_moves)


    # Add edges for the current player's moves
    for move in current_moves:
        from_square_name = chess.square_name(move.from_square)
        to_square_name = chess.square_name(move.to_square)
        G.add_edge(from_square_name, to_square_name)

    # Temporarily switch turns to the opponent
    board.push(chess.Move.null())

    # Get the legal moves for the opponent
    opponent_moves = list(board.legal_moves)


    # Add edges for the opponent's moves
    for move in opponent_moves:
        from_square_name = chess.square_name(move.from_square)
        to_square_name = chess.square_name(move.to_square)
        G.add_edge(from_square_name, to_square_name)

    G.add_node("global")

    # Connect the global node to all squares
    for square in chess.SQUARES:
        square_name = chess.square_name(square)
        G.add_edge("global", square_name)  # Edge from global node to square
        G.add_edge(square_name, "global")  # Edge from square to global node

    # Restore the board to the original state
    # G = G.to_undirected()
    board.pop()

    return G

def square_to_coordinates(square):
    """Convert a square index to board coordinates."""
    row = square // 8 + 1
    col = square % 8 + 1
    return [col, row]

def distance_to_center(square):
    """
    Calculate the distance of a square to the center of the board.
    """
    row, col = divmod(square, 8)
    center_row, center_col = 3.5, 3.5  # Center of the board is at (3.5, 3.5)
    return np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2) / 4.95  # Normalize distance


def piece_to_one_hot(piece):
    """Convert a chess piece to a one-hot encoding including color."""
    pieces = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    one_hot = [0] * 12
    if piece:
        one_hot[pieces[piece.symbol()]] = 1
    return one_hot
def piece_to_value(piece):
    pieces = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
    }
    if piece:
        return [pieces[piece.symbol()]]
    return [0]

def create_node_embeddings(board:chess.Board):
    embeddings = []
    for square in chess.SQUARES:
        coordinates = square_to_coordinates(square)
        piece = board.piece_at(square)
        piece_one_hot = piece_to_one_hot(piece)
        piece_value = piece_to_value(piece)
        center_distance = distance_to_center(square)
        is_attacked_white = int(board.is_attacked_by(True, square))
        is_attacked_black = int(board.is_attacked_by(False, square))
        is_check = int(board.is_check())
        embeddings.append(coordinates+piece_one_hot+piece_value+[center_distance]+[-1**(1-int(board.turn)),is_attacked_white,is_attacked_black,is_check])
    embeddings.append(torch.zeros(20))
    return torch.tensor(embeddings, dtype=torch.float)

def chess_position_to_torch_geometric_data(board):
    # Create the graph using the previous function
    G = chess_position_to_graph(board)

    # Convert the NetworkX graph to edge_index format for torch_geometric
    edge_index = pyg_utils.from_networkx(G).edge_index

    # Create node embeddings
    node_embeddings = create_node_embeddings(board)

    # Create the torch_geometric data object
    data = Data(x=node_embeddings, edge_index=edge_index)

    return data

warnings.filterwarnings("ignore")
device = "cuda:0"
seed = 42 # age of SIPL
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2,aggr="mean"):
        super(GNN, self).__init__()

        # First GNN layer
        self.conv1 = SAGEConv(input_dim, hidden_dim,aggr=aggr)

        # Additional GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim,aggr=aggr))
        # Fully connected layer to produce graph embedding
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128,1)

    def forward(self, data):
        # Extract relevant data from the Data object
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the first GCN layer
        x = F.tanh(self.conv1(x, edge_index))

        # Apply the remaining GCN layers
        for conv in self.convs:
            x = F.tanh(conv(x, edge_index))

        # Pooling to get the graph-level embedding
        x = global_add_pool(x, batch)


        # Final fully connected layer
        x = F.leaky_relu(self.fc1(x),negative_slope=0.2)
        x= self.fc2(x)

        return x

# Example usage:
input_dim = 20  # Example: 2 for location + 12 for one-hot encoding of pieces
hidden_dim = 4*256
output_dim = 128  # Size of the graph embedding

test_model = GNN(input_dim=input_dim, hidden_dim=hidden_dim,num_layers=3,aggr="mean").to(device)
path = "trained model\chess_gnn_model.pth"
test_model.load_state_dict(torch.load(path, map_location=device))

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
    
def get_children(board):
    children = {}
    for move in board.legal_moves:
        child_board = board.copy()
        child_board.push(move)
        children[child_board.fen()]=move
    return children

def evaluate(board):
    # # 'node' is a chess.Board object
    graph_data = chess_position_to_torch_geometric_data(board)
    graph_data = graph_data.to(device)

    with torch.no_grad():
        output = test_model(graph_data)
    return output.item()


fen = chess.STARTING_FEN
board = chess.Board(fen)
print(evaluate(board))

INF = math.inf

def alpha_beta_pruning(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or is_terminal(board):
        if is_terminal(board):
            return game_res(board) * INF, None  # Return evaluation and None as there’s no move
        return evaluate(board), None  # Return evaluation and None for non-terminal but max depth reached

    best_move = None  # Initialize best_move for the top level of recursion

    if maximizing_player:
        max_eval = -INF
        for child, move in get_children(board).items():  # Retrieve both child board and move
            eval, _ = alpha_beta_pruning(chess.Board(child), depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move  # Update best_move with the move leading to the best eval
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval, best_move
    else:
        min_eval = INF
        for child, move in get_children(board).items():
            eval, _ = alpha_beta_pruning(chess.Board(child), depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move  # Update best_move for the minimizing player
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval, best_move

alpha = -INF
beta = INF
DEPTH = 3
# Play against alpha-beta pruning AI
def play_against_ai():
    while not board.is_game_over():
        # Display the board
        print(board)

        # Human's turn
        if board.turn == chess.BLACK:
            print("\nYour move (e.g., e2e4):")
            move = None
            while move not in board.legal_moves:
                try:
                    # Get human move in UCI format (e.g., e2e4)
                    human_move = input("Enter your move: ")
                    move = chess.Move.from_uci(human_move)
                    if move not in board.legal_moves:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid input. Enter moves in UCI format like 'e2e4'.")

            # Make the move on the board
            board.push(move)
        
        # AI's turn
        else:
            print("\nAI is thinking...")
            # Get the best move using alpha-beta pruning
            _, best_move = alpha_beta_pruning(board, DEPTH, -float('inf'), float('inf'), True)
            if best_move:
                board.push(best_move)
                print(f"AI plays: {best_move}")
            else:
                print("AI has no legal moves!")

    # Game over
    print("\nGame over!")
    print(board)
    print("Result:", board.result())

# Run the game
play_against_ai()