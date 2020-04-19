TicTacToe with deep learning 4j
===============================

Tic Tac Toe example with dl4j-1.0.0-beta6
-----------------------------------------

This project provides one main class FeedForwardFourLayerMain.java with adapted net layout compared to the earlier well performing versions with dl4j-0.9.1.
The removed function netIterations(int) has to be replaced with a larger number of epochs. The earlier neural net architectures were no more working with newer dl4j versions.

The new version with only one net architecture did almost reach perfect play: Accuracy, Precision, Recall and F1 Score were all > 0.995 on local test runs. Some more epochs or tuning should make perfect play possible.

Use your favorite IDE or build with mvn to execute the net learning with the present input and labels.

Build parent modul with mvn `~/dl4j$ mvn clean package`

Then run jar with specified training main class: `java -cp target/ch.evolutionsoft.dl.tictactoe-1.1.0-SNAPSHOT-jar-with-dependencies.jar ch.evolutionsoft.example.dl4j.tictactoe.feedforward.FeedForwardFourLayerMain`

A little patience is potentially needed for 2000 training epochs.

Refer to the source or java doc comments for more informations atm.