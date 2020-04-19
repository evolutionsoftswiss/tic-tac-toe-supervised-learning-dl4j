TicTacToe with deep learning 4j
===============================

Tic Tac Toe example with dl4j-1.0.0-beta6
-----------------------------------------

This project provides one main class FeedForwardFourLayerMain.java with adapted net layout compared to the earlier weel performing versions with dl4j-0.9.1.
The removed function netIterations(int) has to be replaced with a larger number of epochs. The earlier neural net architectures were no more working with newer dl4j versions.

Use your favorite IDE or build with mvn to execute the net learning with the present input and labels.

`mvn clean package` followed by `java -cp target/ch.evolutionsoft.dl.tictactoe-1.1.0-SNAPSHOT-jar-with-dependencies.jar ch.evolutionsoft.example.dl4j.tictactoe.feedforward.FeedForwardFourLayerMain`

A little patience is potentially needed for 2000 training epochs.

Refer to the source or java doc comments for more informations atm.