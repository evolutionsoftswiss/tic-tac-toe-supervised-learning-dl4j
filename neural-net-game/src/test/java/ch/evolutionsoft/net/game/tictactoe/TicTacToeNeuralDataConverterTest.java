package ch.evolutionsoft.net.game.tictactoe;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.LinkedList;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static ch.evolutionsoft.net.game.NeuralNetConstants.*;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;

public class TicTacToeNeuralDataConverterTest {

  @Test
  public void convertMiniMaxLabelSingleTest() {
    
    INDArray playground = Nd4j.ones(ONE, COLUMN_NUMBER);
    playground.putScalar(0, FIELD_4, EMPTY_FIELD_VALUE);
    playground.putScalar(0, FIELD_9, EMPTY_FIELD_VALUE);
    playground.putScalar(0, FIELD_2, MIN_PLAYER);
    playground.putScalar(0, FIELD_7, MIN_PLAYER);
    playground.putScalar(0, FIELD_8, MIN_PLAYER);
    
    INDArray miniMaxResult = Nd4j.zeros(ONE, COLUMN_NUMBER);
    miniMaxResult.putScalar(0, FIELD_4, MAX_WIN);
    miniMaxResult.putScalar(0, FIELD_9, MIN_WIN);
    
    List<Pair<INDArray, INDArray>> testSingleList = new LinkedList<Pair<INDArray,INDArray>>();
    testSingleList.add(new Pair<INDArray, INDArray>(playground, miniMaxResult));
    
    List<Pair<INDArray, INDArray>> convertedPairList = TicTacToeNeuralDataConverter.convertMiniMaxLabels(testSingleList);
    INDArray convertedResult = convertedPairList.get(0).getSecond();
    
    assertEquals(convertedResult.getDouble(0, FIELD_4), NET_LOSS);
    assertEquals(convertedResult.getDouble(0, FIELD_9), NET_WIN);
  }

  @Test
  public void convertMiniMaxLabelTwoRowsTest() {
    
    INDArray playground1 = Nd4j.ones(ONE, COLUMN_NUMBER);
    playground1.putScalar(0, FIELD_4, EMPTY_FIELD_VALUE);
    playground1.putScalar(0, FIELD_9, EMPTY_FIELD_VALUE);
    playground1.putScalar(0, FIELD_2, MIN_PLAYER);
    playground1.putScalar(0, FIELD_7, MIN_PLAYER);
    playground1.putScalar(0, FIELD_8, MIN_PLAYER);
    
    INDArray miniMaxResult1 = Nd4j.zeros(ONE, COLUMN_NUMBER);
    miniMaxResult1.putScalar(0, FIELD_4, MAX_WIN);
    miniMaxResult1.putScalar(0, FIELD_9, MIN_WIN);
    
    INDArray playground2 = Nd4j.zeros(ONE, COLUMN_NUMBER);
    INDArray miniMaxResult2 = Nd4j.zeros(ONE, COLUMN_NUMBER);

    
    List<Pair<INDArray, INDArray>> testTwoRowsList = new LinkedList<Pair<INDArray,INDArray>>();
    testTwoRowsList.add(new Pair<INDArray, INDArray>(playground1, miniMaxResult1));
    testTwoRowsList.add(new Pair<INDArray, INDArray>(playground2, miniMaxResult2));
    
    List<Pair<INDArray, INDArray>> convertedPairList = TicTacToeNeuralDataConverter.convertMiniMaxLabels(testTwoRowsList);

    INDArray convertedResult1 = convertedPairList.get(0).getSecond();
    assertEquals(NET_LOSS, convertedResult1.getDouble(0, FIELD_1));
    assertEquals(NET_LOSS, convertedResult1.getDouble(0, FIELD_2));
    assertEquals(NET_LOSS, convertedResult1.getDouble(0, FIELD_3));
    assertEquals(NET_LOSS, convertedResult1.getDouble(0, FIELD_4));
    assertEquals(NET_LOSS, convertedResult1.getDouble(0, FIELD_5));
    assertEquals(NET_LOSS, convertedResult1.getDouble(0, FIELD_6));
    assertEquals(NET_LOSS, convertedResult1.getDouble(0, FIELD_7));
    assertEquals(NET_LOSS, convertedResult1.getDouble(0, FIELD_8));
    assertEquals(NET_WIN, convertedResult1.getDouble(0, FIELD_9));

    
    INDArray convertedResult2 = convertedPairList.get(1).getSecond();
    assertEquals(NET_LOSS, convertedResult2.getDouble(0, FIELD_1));
    assertEquals(NET_LOSS, convertedResult2.getDouble(0, FIELD_2));
    assertEquals(NET_LOSS, convertedResult2.getDouble(0, FIELD_3));
    assertEquals(NET_LOSS, convertedResult2.getDouble(0, FIELD_4));
    assertEquals(NET_LOSS, convertedResult2.getDouble(0, FIELD_5));
    assertEquals(NET_LOSS, convertedResult2.getDouble(0, FIELD_6));
    assertEquals(NET_LOSS, convertedResult2.getDouble(0, FIELD_7));
    assertEquals(NET_LOSS, convertedResult2.getDouble(0, FIELD_8));
    assertEquals(NET_DRAW, convertedResult2.getDouble(0, FIELD_9));
  }
  
  @Test
  public void fastWinMaxPlayerTestAtDifferentFields() {

    
    INDArray playground = Nd4j.zeros(ONE, COLUMN_NUMBER);
    playground.putScalar(0, FIELD_4, MIN_PLAYER);
    playground.putScalar(0, FIELD_9, MAX_PLAYER);
    
    INDArray expectedResultLabels = Nd4j.zeros(ONE, COLUMN_NUMBER);
    expectedResultLabels.putScalar(0, FIELD_3, NET_WIN);
    expectedResultLabels.putScalar(0, FIELD_5, NET_WIN);
    expectedResultLabels.putScalar(0, FIELD_6, NET_WIN);
    
    INDArray miniMaxResult = Nd4j.zeros(ONE, COLUMN_NUMBER);
    miniMaxResult.putScalar(0, FIELD_3, 3 * NET_WIN);
    miniMaxResult.putScalar(0, FIELD_5, 3 * NET_WIN);
    miniMaxResult.putScalar(0, FIELD_6, 3 * NET_WIN);
    
    List<Pair<INDArray, INDArray>> testList = new LinkedList<Pair<INDArray,INDArray>>();
    testList.add(new Pair<INDArray, INDArray>(playground, miniMaxResult));
    
    List<Pair<INDArray, INDArray>> convertedPairList = TicTacToeNeuralDataConverter.convertMiniMaxLabels(testList);
    INDArray convertedResult = convertedPairList.get(0).getSecond();
    
    assertEquals(expectedResultLabels, convertedResult);
  }
}
