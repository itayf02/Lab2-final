This the graphChess lab project.
Make sure that you have an environment the corresponds to the requierments.txt
Make sure that all of the train pgn files are in the same directory as Preprocessing.ipynb
The Preprocessing.ipynb trains the model, if you don't want to retrain, don't run it.
To see the evaluation run evaluate.ipynb. Make sure that the test pgn data are in a directory called "test_data".
If you want to relabel and recreate the stockfish_labels.csv uncomment the lines in cell 10 in evaluation.ipynb.
The trained weights must be in a folder named trained_model.
In order to play against the bot, run play.py and input moves into the terminal.
Moves are in standard algebraic notation (e.g: e2e4). (The bot might take 2 minutes to think, it is preferable you remain on depth 3)
Currently you always play as black.

NOTE: for our graders, we worked together in zoom meetings quite alot, therefor we didn't use commit and pull all that much.
