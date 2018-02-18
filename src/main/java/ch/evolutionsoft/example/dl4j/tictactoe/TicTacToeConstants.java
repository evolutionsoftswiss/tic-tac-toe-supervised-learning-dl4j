package ch.evolutionsoft.example.dl4j.tictactoe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public final class TicTacToeConstants {

  private TicTacToeConstants() {
    // Hide constructor
  }

  /**
   * TicTacToe playground in one row.
   */
  public static final int ROW_NUMBER = 1;
  public static final int COLUMN_NUMBER = 9;
  public static final INDArray EMPTY_PLAYGROUND = Nd4j.zeros(ROW_NUMBER, COLUMN_NUMBER);

  public static final int FIELD_1 = 0;
  public static final int FIELD_2 = 1;
  public static final int FIELD_3 = 2;
  public static final int FIELD_4 = 3;
  public static final int FIELD_5 = 4;
  public static final int FIELD_6 = 5;
  public static final int FIELD_7 = 6;
  public static final int FIELD_8 = 7;
  public static final int FIELD_9 = 8;

  /**
   * TicTacToe playground three channel 3x3 image.
   */
  public static final int IMAGE_SIZE = 3;
  public static final int IMAGE_CHANNELS = 3;

  /**
   * Non empty playground fields ("crosses or circles"), empty is zero.
   */
  public static final double MAX_PLAYER = 1.0;
  public static final double MIN_PLAYER = -1.0;

  public static final INDArray CENTER_FIELD_MOVE = Nd4j.zeros(COLUMN_NUMBER).putScalar(FIELD_5, MAX_PLAYER);
  public static final INDArray LAST_CORNER_FIELD_MOVE = Nd4j.zeros(COLUMN_NUMBER).putScalar(FIELD_9, MAX_PLAYER);

  /**
   * Result labels interpretation. MAX_WIN - 9 > DRAW_VALUE > MIN_WIN + 9 for
   * MiniMax comparison. Any occupied playground field leads to a zero result.
   */
  public static final double MAX_WIN = 10;
  public static final double MIN_WIN = -10;
  public static final int DEPTH_ADVANTAGE = 1;
  public static final double DRAW_VALUE = 0;
  public static final double OCCUPIED = 0;

  public static final double DOUBLE_COMPARISON_EPSILON = 0.01;

}
