package ch.evolutionsoft.net.game.tictactoe;

import static ch.evolutionsoft.net.game.NeuralNetConstants.DOUBLE_COMPARISON_EPSILON;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;

import java.util.LinkedList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

public class TicTacToeGameHelper {
  
  private TicTacToeGameHelper() {
    // Hide constructor
  }

  public static List<Integer> getEmptyFields(INDArray actualPlayGround) {

    List<Integer> emptyFields = new LinkedList<>();
    for (int index = 0; index < COLUMN_COUNT; index++) {

      if (actualPlayGround.getInt(index) == 0) {

        emptyFields.add(index);
      }
    }

    return emptyFields;
  }

  public static boolean noEmptyFieldsLeft(INDArray actualPlayGround) {

    for (int index = 0; index < COLUMN_COUNT; index++) {

      double currentField = actualPlayGround.getDouble(index);
      if (isEmpty(currentField)) {

        return false;
      }
    }

    return true;
  }

  public static boolean allFieldsEmpty(INDArray actualPlayGround) {

    for (int index = 0; index < COLUMN_COUNT; index++) {

      double currentField = actualPlayGround.getDouble(index);
      if (!isEmpty(currentField)) {

        return false;
      }
    }

    return true;
  }

  public static boolean isEmpty(double currentField) {

    return equalsEpsilon(0, currentField, 0.001);
  }
  
  public static double getOpponentPlayer(double currentPlayer) {
    
    if (currentPlayer == MAX_PLAYER) {
      
      return MIN_PLAYER;
    }
    
    return MAX_PLAYER;
  }
  
  public static double getCurrentPlayer(INDArray playground) {
    
    if (isMaxMove(playground)) {
      
      return MAX_PLAYER;
    }
    
    return MIN_PLAYER;
  }

  public static boolean isMaxMove(INDArray playground) {

    int countStones = countStones(playground);
    return countStones % 2 == 0;
  }

  public static int countStones(INDArray playground) {

    int countStones = 0;
    for (int arrayIndex = 0; arrayIndex < COLUMN_COUNT; arrayIndex++) {

      if (!equalsEpsilon(playground.getDouble(arrayIndex), EMPTY_FIELD_VALUE, DOUBLE_COMPARISON_EPSILON)) {

        countStones++;
      }
    }
    return countStones;
  }

  public static int countMaxStones(INDArray playground) {

    int countMaxStones = 0;
    for (int arrayIndex = 0; arrayIndex < COLUMN_COUNT; arrayIndex++) {

      if (equalsEpsilon(playground.getDouble(0, arrayIndex), MAX_PLAYER, DOUBLE_COMPARISON_EPSILON)) {

        countMaxStones++;
      }
    }
    return countMaxStones;
  }

  public static int countMinStones(INDArray playground) {

    int countMinStones = 0;
    for (int arrayIndex = 0; arrayIndex < COLUMN_COUNT; arrayIndex++) {

      if (equalsEpsilon(playground.getDouble(0, arrayIndex), MIN_PLAYER, DOUBLE_COMPARISON_EPSILON)) {

        countMinStones++;
      }
    }
    return countMinStones;
  }

  public static INDArray invertPlayground(INDArray playground) {
    
    INDArray invertedPlayground = playground.dup();

    for (int arrayIndex = 0; arrayIndex < COLUMN_COUNT; arrayIndex++) {

      if (equalsEpsilon(playground.getDouble(0, arrayIndex), MIN_PLAYER, DOUBLE_COMPARISON_EPSILON)) {

        invertedPlayground.putScalar(arrayIndex, MAX_PLAYER);
      
      } else if (equalsEpsilon(playground.getDouble(0, arrayIndex), MAX_PLAYER, DOUBLE_COMPARISON_EPSILON)) {

        invertedPlayground.putScalar(arrayIndex, MIN_PLAYER);
      
      }
    }
    
    return invertedPlayground;
  }
  
  public static boolean equalsEpsilon(double a, double b, double epsilon) {

    if (a == b) {
      return true;
    }

    return Math.abs(a - b) <= epsilon;
  }

  public static boolean hasWon(INDArray actualPlayGround, double player) {

    return horizontalWin(actualPlayGround, player) ||
           verticalWin(actualPlayGround, player) ||
           diagonalWin(actualPlayGround, player);
  }

  protected static boolean horizontalWin(INDArray actualPlayGround, double player) {

    return (actualPlayGround.getDouble(FIELD_1) == player &&
            actualPlayGround.getDouble(FIELD_2) == player &&
            actualPlayGround.getDouble(FIELD_3) == player) ||
           (actualPlayGround.getDouble(FIELD_4) == player &&
            actualPlayGround.getDouble(FIELD_5) == player &&
            actualPlayGround.getDouble(FIELD_6) == player) ||
           (actualPlayGround.getDouble(FIELD_7) == player &&
            actualPlayGround.getDouble(FIELD_8) == player &&
            actualPlayGround.getDouble(FIELD_9) == player);
  }

  protected static boolean diagonalWin(INDArray actualPlayGround, double player) {

    return (actualPlayGround.getDouble(FIELD_1) == player &&
            actualPlayGround.getDouble(FIELD_5) == player &&
            actualPlayGround.getDouble(FIELD_9) == player) ||
           (actualPlayGround.getDouble(FIELD_3) == player &&
            actualPlayGround.getDouble(FIELD_5) == player &&
            actualPlayGround.getDouble(FIELD_7) == player);
  }

  protected static boolean verticalWin(INDArray actualPlayGround, double player) {

    return (actualPlayGround.getDouble(FIELD_1) == player &&
            actualPlayGround.getDouble(FIELD_4) == player &&
            actualPlayGround.getDouble(FIELD_7) == player) ||
           (actualPlayGround.getDouble(FIELD_2) == player &&
            actualPlayGround.getDouble(FIELD_5) == player &&
            actualPlayGround.getDouble(FIELD_8) == player) ||
           (actualPlayGround.getDouble(FIELD_3) == player &&
            actualPlayGround.getDouble(FIELD_6) == player &&
            actualPlayGround.getDouble(FIELD_9) == player);
  }
}
