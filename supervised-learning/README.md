TicTacToe with deep learning 4j
===============================

Tic Tac Toe example with dl4j-1.0.0-beta7
-----------------------------------------

### Overview
This project provides the main class FeedForwardFourLayerMain with two hidden dense layers and totally 2729 parameters. That variation uses a balanced DataSet input with only a subset of each possible labels per epoch.

Another main class ConvolutionalNetMain is executing the supervised TicTacToe learning with a very small convolutional residual architecture. ConvolutionalNetMain uses an EarlyStoppingTrainer with the whole unbalanced DataSet. A lot of epochs not leading to a better net performance are discarded.

Those two net architectures with the specified training did almost reach perfect play: Accuracy, Precision, Recall and F1 Score were all > 0.995 on local test runs. Some more epochs or tuning should make perfect play possible.

Use your favorite IDE or build with mvn to execute the net learning with the present input and labels.

On latest update I've added AVX/AVX-2 support under linux to get potentially a better performance. You need to adapt the classifier in the pom.xml's of the two submodules to build for MacOS or Windows. You could also remove the AVX/AVX-2 support completely to get the build OS-independent: To do this exclude the following dependency from both pom.xml's in the submodules:


### Build instructions
Build parent modul with mvn `~/dl4j$ mvn clean package`

Then run jar with specified training main class: `java -cp supervised-learning/target/ch.evolutionsoft.dl.tictactoe-1.1.1-SNAPSHOT-jar-with-dependencies.jar ch.evolutionsoft.example.dl4j.tictactoe.feedforward.FeedForwardFourLayerMain`

or `java -cp supervised-learning/target/ch.evolutionsoft.dl.tictactoe-1.1.1-SNAPSHOT-jar-with-dependencies.jar ch.evolutionsoft.example.dl4j.tictactoe.convolutional.ConvolutionalNetMain` for the convolutional version respectively.

A little patience is potentially needed to perform all training epochs.
