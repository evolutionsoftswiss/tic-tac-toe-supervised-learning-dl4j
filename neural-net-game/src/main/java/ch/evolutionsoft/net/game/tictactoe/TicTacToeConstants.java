package ch.evolutionsoft.net.game.tictactoe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public final class TicTacToeConstants {

  private TicTacToeConstants() {
    // Hide constructor
  }

  /**
   * TicTacToe playground in one row.
   */
  public static final int ROW_COUNT = 1;
  public static final int COLUMN_COUNT = 9;
  public static final INDArray EMPTY_PLAYGROUND = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);

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
  public static final int IMAGE_POINTS = 9;
  
  public static final int PLAYER_CHANNEL = 0;
  public static final int MAX_PLAYER_CHANNEL = 1;
  public static final int MIN_PLAYER_CHANNEL = 2;
  
  public static final double EMPTY_IMAGE_POINT = 0;
  public static final double OCCUPIED_IMAGE_POINT = 1;
  
  public static final INDArray ZEROS_PLAYGROUND_IMAGE = Nd4j.zeros(1, IMAGE_SIZE, IMAGE_SIZE);
  public static final INDArray ONES_PLAYGROUND_IMAGE = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
  public static final INDArray MINUS_ONES_PLAYGROUND_IMAGE = ZEROS_PLAYGROUND_IMAGE.sub(1);
  
  public static final INDArray EMPTY_CONVOLUTIONAL_PLAYGROUND = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
  static {
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(PLAYER_CHANNEL, ONES_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(MAX_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(MIN_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
  }

  /**
   * Non empty playground fields ("crosses or circles"), empty is zero.
   */
  public static final double MAX_PLAYER = 1.0;
  public static final double MIN_PLAYER = -1.0;
  public static final double EMPTY_FIELD_VALUE = 0.0;

  public static final INDArray CENTER_FIELD_MOVE = Nd4j.zeros(1, COLUMN_COUNT).putScalar(FIELD_5, MAX_PLAYER);
  public static final INDArray LAST_CORNER_FIELD_MOVE = Nd4j.zeros(1, COLUMN_COUNT).putScalar(FIELD_9, MAX_PLAYER);

  /**
   * Result labels interpretation. MAX_WIN - 9 > DRAW_VALUE > MIN_WIN + 9 for
   * MiniMax comparison. Any occupied playground field leads to a zero result.
   */
  public static final int MAX_WIN = 10;
  public static final int MIN_WIN = -10;
  public static final int DEPTH_ADVANTAGE = 1;
  public static final int MINIMAX_DRAW_VALUE = 0;
  public static final double OCCUPIED = 0;

  /**
   * Result labels for neural nets 0 corresponds to a field that we don't want to play
   * 1 to a field that we want to play, leading to all possible results from the current
   * player perspective.
   */
  public static final double NET_WIN = 1.0;
  public static final double NET_DRAW = 1.0;
  public static final double NET_LOSS = 1.0;
  
  public static final int SMALL_CAPACITY = 10;
}
