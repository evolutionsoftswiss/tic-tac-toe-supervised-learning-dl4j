TicTacToe with deep learning 4j
===============================

Tic Tac Toe example with dl4j-1.0.0-beta6
-----------------------------------------

### Overview
This project provides the main class FeedForwardFourLayerMain with adapted net layout compared to the earlier well performing versions with dl4j-0.9.1. That variation uses a balanced DataSet input with only a subset of each possible labels per epoch.

Another main class ConvolutionalNetMain is executing the supervised TicTacToe learning with a very small convolutional residual architecture. ConvolutionalNetMain uses an EarlyStoppingTrainer with the whole unbalanced DataSet. A lot of epochs not leading to a better net performance are discarded.

The new version with only two net architectures did almost reach perfect play: Accuracy, Precision, Recall and F1 Score were all > 0.995 on local test runs. Some more epochs or tuning should make perfect play possible.

Use your favorite IDE or build with mvn to execute the net learning with the present input and labels.

### Historical note
The removed function netIterations(int) from dl4j-0.9.1 had to be replaced with a larger number of epochs principally. Also the earlier neural net architectures and configurations were no more working with newer dl4j versions.

### Build instructions
Build parent modul with mvn `~/dl4j$ mvn clean package`

Then run jar with specified training main class: `java -cp supervised-learning/target/ch.evolutionsoft.dl.tictactoe-1.1.0-SNAPSHOT-jar-with-dependencies.jar ch.evolutionsoft.example.dl4j.tictactoe.feedforward.FeedForwardFourLayerMain`

or `java -cp supervised-learning/target/ch.evolutionsoft.dl.tictactoe-1.1.0-SNAPSHOT-jar-with-dependencies.jar ch.evolutionsoft.example.dl4j.tictactoe.convolutional.ConvolutionalNetMain` for the convolutional version respectively.

A little patience is potentially needed to perform all training epochs.
