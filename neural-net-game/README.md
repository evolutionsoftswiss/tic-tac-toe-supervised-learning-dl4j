# Neural Net game Module

## Tic Tac Toe

This module provides some classes to support learning Tic Tac Toe with neural networks.

It provides two classes with constants and a TicTacToeGameHelper with some functions reflecting the game logic.

The TicTacToeNeuralDataConverter contains methods to rearrange the labeled Inputs for a neural network.
It generates network labels from MiniMax tree search results. The playground is represented as one dimensional array with 9 columns, e.g. for dense nets.
On the other hand it also generates image like playground representations with three channels, one for empty fields, and two for each player. That representation is like a 3x3 pixel image with three channels and has thus the shape (3, 3, 3).

The TicTacToeMiniMaxGenerator can be used to regenerate the move space.