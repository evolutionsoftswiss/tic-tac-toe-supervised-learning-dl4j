package ch.evolutionsoft.net.game.tictactoe;

import static ch.evolutionsoft.net.game.NeuralNetConstants.DOUBLE_COMPARISON_EPSILON;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.NeuralDataHelper;

/**
 * Generates TicTacToe playgrounds with result labels defined by several
 * constants.
 * 
 * The idea is to generate all possible positions by a MiniMax tree traversal.
 * The playground arrays and corresponding values are stored as pairs in a list.
 * 
 * The initial generation should be ready in a few seconds. Duplicate removal is
 * rather slow.
 */
public class TicTacToeMiniMaxGenerator {

  private static final Logger logger = LoggerFactory.getLogger(TicTacToeMiniMaxGenerator.class);

  protected boolean keepDuplicates = false;

  protected List<Pair<INDArray, INDArray>> allPlaygroundsResults = new ArrayList<>();
  
  protected int bestFoundSearchMove = -1;

  public static void main(String[] arguments) {

    TicTacToeMiniMaxGenerator data = new TicTacToeMiniMaxGenerator();
    logger.info("Data Processing Started : {}", new Date());

    data.searchInitial();
    logger.info("All possible game state sequences generated, Finished At : {}", new Date());

    data.allPlaygroundsResults = data.removeDuplicates();
    logger.info("Unique game states filteres, Finished At : {}", new Date());

    NeuralDataHelper.writeData(data.allPlaygroundsResults);
    logger.info("File generation completed : at {}", new Date());
  }

  public void searchInitial() {

    this.max(EMPTY_PLAYGROUND, 0);
  }

  public int searchCurrent(INDArray currentPlayground) {

    double currentPlayer = TicTacToeGameHelper.getCurrentPlayer(currentPlayground);
    
    int fieldsOccupied = TicTacToeGameHelper.countStones(currentPlayground);
    
    if (currentPlayer == MAX_PLAYER) {
      
      int currentValue = Integer.MIN_VALUE;
      for (int currentMove : TicTacToeGameHelper.getEmptyFields(currentPlayground)) {
        
        INDArray newPlayground = performMove(currentPlayground, currentMove, currentPlayer);
        
        int newValue = searchMin(newPlayground, Integer.MIN_VALUE, Integer.MAX_VALUE, fieldsOccupied + 1);
        
        if (newValue > currentValue) {
          
          currentValue = newValue;
          this.bestFoundSearchMove = currentMove;
        }
      }
      
      return currentValue;

    } else {

      int currentValue = Integer.MAX_VALUE;
      for (int currentMove : TicTacToeGameHelper.getEmptyFields(currentPlayground)) {
        
        INDArray newPlayground = performMove(currentPlayground, currentMove, currentPlayer);
        
        int newValue = searchMax(newPlayground, Integer.MIN_VALUE, Integer.MAX_VALUE, fieldsOccupied + 1);
        
        if (newValue < currentValue) {
          
          currentValue = newValue;
          this.bestFoundSearchMove = currentMove;
        }
      }
      
      return currentValue;
    }
  }

  public List<Pair<INDArray, INDArray>> removeDuplicates() {

    Set<INDArray> presentPlaygrounds = new HashSet<>();

    List<Pair<INDArray, INDArray>> uniquePlaygroundsResults = new ArrayList<>();

    for (Pair<INDArray, INDArray> currentPair : allPlaygroundsResults) {

      INDArray currentPlayground = currentPair.getFirst();
      if (!alreadyPresent(presentPlaygrounds, currentPlayground)) {

        presentPlaygrounds.add(currentPlayground);
        uniquePlaygroundsResults.add(currentPair);
      }
    }

    this.allPlaygroundsResults = uniquePlaygroundsResults;

    return uniquePlaygroundsResults;
  }

  public List<Pair<INDArray, INDArray>> getGeneratedPlaygroundsLabels() {

    return allPlaygroundsResults;
  }

  public int searchMax(INDArray currentPlayground, int alpha, int beta, int depth) {

    if (TicTacToeGameHelper.hasWon(currentPlayground, MIN_PLAYER)) {

      return MIN_WIN + depth;

    } else if (TicTacToeGameHelper.noEmptyFieldsLeft(currentPlayground)) {

      return MINIMAX_DRAW_VALUE;

    }

    int currentValue = MIN_WIN;
    for (int currentMove = 0; currentMove < COLUMN_COUNT; currentMove++) {

      if (currentPlayground.getDouble(currentMove) == 0) {

        INDArray newPlayground = performMove(currentPlayground, currentMove, MAX_PLAYER);

        currentValue = Math.max(currentValue, searchMin(newPlayground, alpha, beta, depth + 1));

        if (currentValue > beta) {
            return currentValue;
        }

        alpha = Math.max(alpha, currentValue);
      }

    }

    return currentValue;
  }

  public int searchMin(INDArray currentPlayground, int alpha, int beta, int depth) {

    if (TicTacToeGameHelper.hasWon(currentPlayground, MAX_PLAYER)) {

      return MAX_WIN - depth;

    } else if (TicTacToeGameHelper.noEmptyFieldsLeft(currentPlayground)) {

      return MINIMAX_DRAW_VALUE;

    }

    int currentValue = MAX_WIN;
    for (int currentMove = 0; currentMove < COLUMN_COUNT; currentMove++) {

      if (currentPlayground.getDouble(currentMove) == 0) {

        INDArray newPlayground = performMove(currentPlayground, currentMove, MIN_PLAYER);

        currentValue = Math.min(currentValue, searchMax(newPlayground, alpha, beta, depth + 1));
        
        if (currentValue < alpha) {
             return currentValue;
        }
      
        beta = Math.min(currentValue, beta);
      }
    }

    return currentValue;
  }

  public int getBestFoundSearchMove() {

    return bestFoundSearchMove;
  }

  protected INDArray performMove(INDArray currentPlayground, int currentMove, double currentPlayer) {

    INDArray newPlayground = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);
    Nd4j.copy(currentPlayground, newPlayground);

    newPlayground.putScalar(0, currentMove, currentPlayer);
    return newPlayground;
  }

  protected int max(INDArray currentPlayground, int depth) {

    if (TicTacToeGameHelper.hasWon(currentPlayground, MIN_PLAYER)) {

      return MIN_WIN + depth;

    } else if (TicTacToeGameHelper.noEmptyFieldsLeft(currentPlayground)) {

      return MINIMAX_DRAW_VALUE;

    }

    INDArray currentResults = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);
    int currentValue = MIN_WIN;
    for (int currentMove = 0; currentMove < COLUMN_COUNT; currentMove++) {

      if (currentPlayground.getDouble(currentMove) == 0) {

        INDArray newPlayground = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);
        Nd4j.copy(currentPlayground, newPlayground);

        newPlayground.putScalar(0, currentMove, MAX_PLAYER);

        currentValue = Math.max(currentValue, min(newPlayground, depth + 1));

        currentResults.putScalar(0, currentMove, currentValue);

      } else {

        currentResults.putScalar(0, currentMove, OCCUPIED);
      }

    }

    allPlaygroundsResults.add(new Pair<INDArray, INDArray>(currentPlayground, currentResults));

    return currentValue;

  }

  protected int min(INDArray currentPlayground, int depth) {

    if (TicTacToeGameHelper.hasWon(currentPlayground, MAX_PLAYER)) {

      return MAX_WIN - depth;

    } else if (TicTacToeGameHelper.noEmptyFieldsLeft(currentPlayground)) {

      return MINIMAX_DRAW_VALUE;

    }

    INDArray currentResults = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);
    int currentValue = MAX_WIN;
    for (int currentMove = 0; currentMove < COLUMN_COUNT; currentMove++) {

      if (currentPlayground.getDouble(currentMove) == 0) {

        INDArray newPlayground = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);
        Nd4j.copy(currentPlayground, newPlayground);

        newPlayground.putScalar(0, currentMove, MIN_PLAYER);

        currentValue = Math.min(currentValue, max(newPlayground, depth + 1));

        currentResults.putScalar(0, currentMove, currentValue);

      } else {

        currentResults.putScalar(0, currentMove, OCCUPIED);
      }
    }

    allPlaygroundsResults.add(new Pair<INDArray, INDArray>(currentPlayground, currentResults));

    return currentValue;
  }

  protected boolean alreadyPresent(Set<INDArray> uniquePlaygrounds, INDArray playground) {

    for (INDArray currentPlayground : uniquePlaygrounds) {

      if (currentPlayground.equalsWithEps(playground, DOUBLE_COMPARISON_EPSILON)) {

        return true;
      }
    }
    return false;
  }

  protected INDArray createVector(double scalarValue) {

    INDArray allDraw = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);

    for (int index = 0; index < COLUMN_COUNT; index++) {

      allDraw.putScalar(0, index, scalarValue);
    }

    return allDraw;
  }
}
