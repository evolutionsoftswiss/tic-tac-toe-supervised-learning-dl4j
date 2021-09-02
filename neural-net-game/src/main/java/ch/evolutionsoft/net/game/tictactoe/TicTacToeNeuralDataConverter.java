package ch.evolutionsoft.net.game.tictactoe;

import static ch.evolutionsoft.net.game.NeuralNetConstants.DOUBLE_COMPARISON_EPSILON;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeGameHelper.*;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.net.game.NeuralNetConstants;

public class TicTacToeNeuralDataConverter {

  public static final double SMALLEST_MAX_WIN = 1;
  public static final double BIGGEST_MIN_WIN = -1;

  private TicTacToeNeuralDataConverter() {
    // Hide constructor
  }

  public static List<Pair<INDArray, INDArray>> generateMultiClassLabelsConvolutional(List<Pair<INDArray, INDArray>> allPlaygroundsResults) {

    List<Pair<INDArray, INDArray>> convertedLabels = convertMultiMiniMaxLabels(allPlaygroundsResults);
    
    List<Pair<INDArray, INDArray>> resultList = new LinkedList<>();

    for (int index = 0; index < allPlaygroundsResults.size(); index++) {

      INDArray playgroundArray = convertedLabels.get(index).getFirst();

      INDArray playgroundImage = TicTacToeNeuralDataConverter.convertTo3x3Image(playgroundArray);
      INDArray playgroundImage4dRow = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
      playgroundImage4dRow.putRow(0, playgroundImage);

      resultList.add(new Pair<INDArray, INDArray>(playgroundImage4dRow, convertedLabels.get(index).getSecond()));
    }

    return resultList;
  }

  public static List<Pair<INDArray, INDArray>> convertMiniMaxPlaygroundLabelsToConvolutionalData(
      List<Pair<INDArray, INDArray>> allPlaygroundsResults) {

    List<Pair<INDArray, INDArray>> convertedLabels = convertMiniMaxLabels(allPlaygroundsResults);
    List<Pair<INDArray, INDArray>> resultList = new LinkedList<>();

    for (int index = 0; index < allPlaygroundsResults.size(); index++) {

      INDArray playgroundArray = convertedLabels.get(index).getFirst();

      INDArray playgroundImage = TicTacToeNeuralDataConverter.convertTo3x3Image(playgroundArray);
      INDArray playgroundImage4dRow = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
      playgroundImage4dRow.putRow(0, playgroundImage);

      resultList.add(new Pair<INDArray, INDArray>(playgroundImage4dRow, convertedLabels.get(index).getSecond()));
    }

    return resultList;
  }

  public static Pair<INDArray, INDArray> stackFeedForwardPlaygroundLabels(
      List<Pair<INDArray, INDArray>> adaptedPlaygroundsLabels) {

    int playgroundsLabelsSize = adaptedPlaygroundsLabels.size();
    INDArray stackedPlaygrounds = Nd4j.zeros(playgroundsLabelsSize, COLUMN_COUNT);
    INDArray stackedLabels = Nd4j.zeros(playgroundsLabelsSize, COLUMN_COUNT);

    for (int index = 0; index < playgroundsLabelsSize; index++) {

      INDArray currentPlayground = adaptedPlaygroundsLabels.get(index).getFirst();
      stackedPlaygrounds.putRow(index, currentPlayground);

      INDArray currentLabel = adaptedPlaygroundsLabels.get(index).getSecond();
      stackedLabels.putRow(index, currentLabel);
    }

    return new Pair<>(stackedPlaygrounds, stackedLabels);
  }

  public static Pair<INDArray, INDArray> stackConvolutionalPlaygroundLabels(
      List<Pair<INDArray, INDArray>> adaptedPlaygroundsLabels) {

    int playgroundsLabelsSize = adaptedPlaygroundsLabels.size();
    INDArray stackedPlaygrounds = Nd4j.zeros(playgroundsLabelsSize, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    INDArray stackedLabels = Nd4j.zeros(playgroundsLabelsSize, COLUMN_COUNT);

    for (int index = 0; index < playgroundsLabelsSize; index++) {

      INDArray currentPlayground = adaptedPlaygroundsLabels.get(index).getFirst();
      stackedPlaygrounds.putRow(index, currentPlayground);

      INDArray currentLabel = adaptedPlaygroundsLabels.get(index).getSecond();
      stackedLabels.putRow(index, currentLabel);
    }

    return new Pair<>(stackedPlaygrounds, stackedLabels);
  }

  public static INDArray convertTo3x3Image(INDArray playgroundArray) {

    INDArray playerArray = Nd4j.zeros(IMAGE_SIZE, IMAGE_SIZE);
    INDArray maxArray = Nd4j.zeros(IMAGE_SIZE, IMAGE_SIZE);
    INDArray minArray = Nd4j.zeros(IMAGE_SIZE, IMAGE_SIZE);

    int occupiedFields = 0;

    for (int row = 0; row < IMAGE_SIZE; row++) {
      
      for (int column = 0; column < IMAGE_SIZE; column++) {

        int flatIndex = IMAGE_SIZE * row + column;
        double playgroundValue = playgroundArray.getDouble(flatIndex);

        if (playgroundValue == MIN_PLAYER) {

          minArray.putScalar(row, column, 1);
          occupiedFields++;

        } else if (playgroundValue == MAX_PLAYER) {

          maxArray.putScalar(row, column, 1);
          occupiedFields++;
        }
      }
      if (0 == occupiedFields % 2) {
        
        playerArray = Nd4j.ones(IMAGE_SIZE, IMAGE_SIZE);
      
      } else {
        
        playerArray = Nd4j.ones(IMAGE_SIZE, IMAGE_SIZE).mul(-1);
      }
    }

    INDArray playgroundImage = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    playgroundImage.putRow(0, playerArray);
    playgroundImage.putRow(1, maxArray);
    playgroundImage.putRow(2, minArray);

    return playgroundImage;
  }

  public static List<Pair<INDArray, INDArray>> convertMultiMiniMaxLabels(
      List<Pair<INDArray, INDArray>> allPlaygroundsResults) {

    List<Pair<INDArray, INDArray>> adaptedPlaygroundsLabels = new LinkedList<>();

    for (Pair<INDArray, INDArray> currentPair : allPlaygroundsResults) {
      
      INDArray currentPlayground = currentPair.getFirst();

      INDArray currentResult = currentPair.getSecond();
      INDArray adaptedResult = convertMiniMaxResultToMultiClassNetLabel(currentPlayground, currentResult);

      adaptedPlaygroundsLabels.add(new Pair<>(currentPlayground, adaptedResult));

    }

    return adaptedPlaygroundsLabels;
  }

  public static List<Pair<INDArray, INDArray>> convertMiniMaxLabels(
      List<Pair<INDArray, INDArray>> allPlaygroundsResults) {

    List<Pair<INDArray, INDArray>> adaptedPlaygroundsLabels = new ArrayList<>(NeuralNetConstants.MEDIUM_CAPACITY);

    for (int index = 0; index < allPlaygroundsResults.size(); index++) {

      INDArray currentPlayground = allPlaygroundsResults.get(index).getFirst();

      INDArray currentResult = allPlaygroundsResults.get(index).getSecond();
      INDArray adaptedResult = convertMiniMaxResultToBinaryNetLabel(currentPlayground, currentResult);
      
      adaptedPlaygroundsLabels.add(new Pair<>(currentPlayground, adaptedResult));
    }

    return adaptedPlaygroundsLabels;
  }

  protected static INDArray convertMiniMaxResultToMultiClassNetLabel(INDArray currentPlayground, INDArray currentResult) {

    int numberOfDrawMoves = 0;
    int numberOfMaxWins = 0;
    int numberOfMinWins = 0;
    for (int arrayIndex = 0; arrayIndex < COLUMN_COUNT; arrayIndex++) {

      if (equalsEpsilon(currentPlayground.getDouble(arrayIndex), EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON) &&
          equalsEpsilon(currentResult.getDouble(arrayIndex), MINIMAX_DRAW_VALUE, DOUBLE_COMPARISON_EPSILON)) {

        numberOfDrawMoves++;

      } else if (equalsEpsilon(currentPlayground.getDouble(arrayIndex), EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON) &&
          currentResult.getDouble(arrayIndex) >= SMALLEST_MAX_WIN) {

        numberOfMaxWins++;

      } else if (equalsEpsilon(currentPlayground.getDouble(0, arrayIndex), EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON) &&
          currentResult.getDouble(arrayIndex) <= BIGGEST_MIN_WIN) {

        numberOfMinWins++;
      
      }
    }

    INDArray adaptedResult = null;

    if (isMaxMove(currentPlayground) && numberOfMaxWins > 0) {

      adaptedResult = handleMultiMaxWinPosition(currentResult, numberOfMaxWins);

    } else if (!isMaxMove(currentPlayground) && numberOfMinWins > 0) {

      adaptedResult = handleMultiMinWinPosition(currentResult, numberOfMinWins);

    } else if (numberOfDrawMoves > 0) {

      adaptedResult = handleMultiDrawPosition(currentPlayground, currentResult);

    } else if (isMaxMove(currentPlayground) && numberOfMinWins > 0) {

      adaptedResult = handleMultiLossPosition(currentPlayground);
    
    } else if (!isMaxMove(currentPlayground) && numberOfMaxWins > 0) {

      adaptedResult = handleMultiLossPosition(currentPlayground);
    }

    return adaptedResult;
  }

  protected static INDArray convertMiniMaxResultToBinaryNetLabel(INDArray currentPlayground, INDArray currentResult) {

    int numberOfDrawMoves = 0;
    int numberOfMaxWins = 0;
    int numberOfMinWins = 0;
    double currentFastestMaxWin = SMALLEST_MAX_WIN;
    double currentFastestMinWin = BIGGEST_MIN_WIN;
    int bestMaxIndex = -1;
    int bestMinIndex = -1;
    for (int arrayIndex = 0; arrayIndex < COLUMN_COUNT; arrayIndex++) {

      double playgroundOccupation = currentPlayground.getDouble(arrayIndex);
      double upcomingFieldResult = currentResult.getDouble(arrayIndex);
      
      if (equalsEpsilon(playgroundOccupation, EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON) &&
          equalsEpsilon(upcomingFieldResult, MINIMAX_DRAW_VALUE, DOUBLE_COMPARISON_EPSILON)) {

        numberOfDrawMoves++;

      } else if (equalsEpsilon(playgroundOccupation, EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON) &&
          SMALLEST_MAX_WIN <= upcomingFieldResult) {

        numberOfMaxWins++;
        if (currentFastestMaxWin < upcomingFieldResult ||
            bestMaxIndex == -1) {

          currentFastestMaxWin = upcomingFieldResult;
          bestMaxIndex = arrayIndex;
        
        } 

      } else if (equalsEpsilon(playgroundOccupation, EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON) &&
          BIGGEST_MIN_WIN >= upcomingFieldResult) {

        numberOfMinWins++;
        if (currentFastestMinWin > upcomingFieldResult ||
            bestMinIndex == -1) {

          currentFastestMinWin = upcomingFieldResult;
          bestMinIndex = arrayIndex;

        }
      }
    }

    INDArray adaptedResult;

    if (isMaxMove(currentPlayground) && numberOfMaxWins > 0) {

      adaptedResult = Nd4j.zeros(COLUMN_COUNT).putScalar(bestMaxIndex, NET_WIN);

    } else if (!isMaxMove(currentPlayground) && numberOfMinWins > 0) {

      adaptedResult = Nd4j.zeros(COLUMN_COUNT).putScalar(bestMinIndex, NET_WIN);

    } else if (numberOfDrawMoves > 0) {

      adaptedResult = handleDrawPosition(currentPlayground, currentResult);

    } else {
      
      adaptedResult = handleLossPosition(currentPlayground);
    }

    return adaptedResult;
  }

  protected static INDArray handleMultiMaxWinPosition(INDArray currentResult, int maxWins) {

    INDArray adaptedResult = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);

    int winFieldsFound = 0;
    double fastestWinFieldValue = SMALLEST_MAX_WIN - DEPTH_ADVANTAGE;
    for (int arrayIndex = 0; arrayIndex < 9 && winFieldsFound < maxWins; arrayIndex++) {

      double currentWinFieldValue = currentResult.getDouble(arrayIndex);

      if (currentWinFieldValue > fastestWinFieldValue) {

        fastestWinFieldValue = currentWinFieldValue;

        adaptedResult = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);
        adaptedResult.putScalar(arrayIndex, NET_WIN);
        winFieldsFound++;

      } else if (currentWinFieldValue > NET_DRAW && equalsEpsilon(currentWinFieldValue, fastestWinFieldValue, DOUBLE_COMPARISON_EPSILON) ) {

        adaptedResult.putScalar(arrayIndex, NET_WIN);
        winFieldsFound++;
      }
    }
    return adaptedResult;
  }

  protected static INDArray handleMultiMinWinPosition(INDArray currentResult, int minWins) {

    INDArray adaptedResult = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);

    int winFieldsFound = 0;
    double fastestWinFieldValue = BIGGEST_MIN_WIN + DEPTH_ADVANTAGE;
    for (int arrayIndex = 0; arrayIndex < 9 && winFieldsFound < minWins; arrayIndex++) {

      double currentWinFieldValue = currentResult.getDouble(arrayIndex);

      if (currentWinFieldValue < fastestWinFieldValue) {

        adaptedResult = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);

        fastestWinFieldValue = currentWinFieldValue;
        adaptedResult.putScalar(arrayIndex, NET_WIN);

        winFieldsFound++;

      } else if (currentWinFieldValue < NET_DRAW && equalsEpsilon(currentWinFieldValue, fastestWinFieldValue, DOUBLE_COMPARISON_EPSILON) ) {

        adaptedResult.putScalar(arrayIndex, NET_WIN);
        winFieldsFound++;
      }
    }
    return adaptedResult;
  }

  protected static INDArray handleDrawPosition(INDArray currentPlayground, INDArray currentResult) {

    // Take the first field found leading to a draw
    INDArray adaptedResult = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);
    boolean drawFieldFound = false;

    for (int arrayIndex = 0; arrayIndex < 9 && !drawFieldFound; arrayIndex++) {

      if (equalsEpsilon(currentPlayground.getDouble(arrayIndex), EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON) &&
          equalsEpsilon(currentResult.getDouble(arrayIndex), MINIMAX_DRAW_VALUE, DOUBLE_COMPARISON_EPSILON)) {

        drawFieldFound = true;
        adaptedResult.putScalar(0, arrayIndex, NET_DRAW);
      }
    }

    return adaptedResult;
  }

  protected static INDArray handleMultiDrawPosition(INDArray currentPlayground, INDArray currentResult) {

    INDArray adaptedResult = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);

    for (int arrayIndex = 0; arrayIndex < 9; arrayIndex++) {

      if (equalsEpsilon(currentPlayground.getDouble(arrayIndex), EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON) &&
          equalsEpsilon(currentResult.getDouble(arrayIndex), MINIMAX_DRAW_VALUE, DOUBLE_COMPARISON_EPSILON)) {

        adaptedResult.putScalar(arrayIndex, NET_DRAW);
      }
    }

    return adaptedResult;
  }

  protected static INDArray handleLossPosition(INDArray currentPlayground) {

    // Take the first found empty field leading to loss
    boolean lossFieldFound = false;
    INDArray adaptedResult = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);

    for (int arrayIndex = 0; arrayIndex < 9 && !lossFieldFound; arrayIndex++) {

      if (equalsEpsilon(currentPlayground.getDouble(arrayIndex), EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON)) {

        lossFieldFound = true;
        adaptedResult.putScalar(0, arrayIndex, NET_LOSS);
      }
    }

    return adaptedResult;
  }

  protected static INDArray handleMultiLossPosition(INDArray currentPlayground) {

    double lossValue = 1.0;

    INDArray adaptedResult = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);

    for (int arrayIndex = 0; arrayIndex < 9; arrayIndex++) {

      if (equalsEpsilon(currentPlayground.getDouble(arrayIndex), EMPTY_FIELD_VALUE,
          DOUBLE_COMPARISON_EPSILON)) {

        adaptedResult.putScalar(arrayIndex, lossValue);
      }
    }

    return adaptedResult;
  }

}
