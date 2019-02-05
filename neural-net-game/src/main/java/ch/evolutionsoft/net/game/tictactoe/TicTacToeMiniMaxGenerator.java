package ch.evolutionsoft.net.game.tictactoe;

import static ch.evolutionsoft.net.game.NeuralNetConstants.DOUBLE_COMPARISON_EPSILON;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.LinkedList;
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

  protected List<Pair<INDArray, INDArray>> allPlaygroundsResults = new LinkedList<>();

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

  protected double max(INDArray currentPlayground, int depth) {

    if (TicTacToeGameHelper.hasWon(currentPlayground, MIN_PLAYER)) {

      return MIN_WIN + depth;

    } else if (TicTacToeGameHelper.noEmptyFieldsLeft(currentPlayground)) {

      return DRAW_VALUE;

    }

    INDArray currentResults = Nd4j.createUninitialized(ROW_NUMBER, COLUMN_NUMBER);
    double currentValue = MIN_WIN;
    for (int currentMove = 0; currentMove < COLUMN_NUMBER; currentMove++) {

      if (currentPlayground.getDouble(currentMove) == 0) {

        INDArray newPlayground = Nd4j.createUninitialized(ROW_NUMBER, COLUMN_NUMBER);
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

  protected double min(INDArray currentPlayground, int depth) {

    if (TicTacToeGameHelper.hasWon(currentPlayground, MAX_PLAYER)) {

      return MAX_WIN - depth;

    } else if (TicTacToeGameHelper.noEmptyFieldsLeft(currentPlayground)) {

      return DRAW_VALUE;

    }

    INDArray currentResults = Nd4j.createUninitialized(ROW_NUMBER, COLUMN_NUMBER);
    double currentValue = MAX_WIN;
    for (int currentMove = 0; currentMove < COLUMN_NUMBER; currentMove++) {

      if (currentPlayground.getDouble(currentMove) == 0) {

        INDArray newPlayground = Nd4j.createUninitialized(ROW_NUMBER, COLUMN_NUMBER);
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

    INDArray allDraw = Nd4j.createUninitialized(ROW_NUMBER, COLUMN_NUMBER);

    for (int index = 0; index < COLUMN_NUMBER; index++) {

      allDraw.putScalar(0, index, scalarValue);
    }

    return allDraw;
  }
}
