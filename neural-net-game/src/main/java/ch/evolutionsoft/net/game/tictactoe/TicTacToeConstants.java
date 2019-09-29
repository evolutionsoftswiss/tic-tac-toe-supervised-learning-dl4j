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
  public static final int IMAGE_POINTS = 9;
  public static final int[] IMAGE_DIMENSION = new int[] {IMAGE_SIZE, IMAGE_SIZE};
  
  public static final int EMPTY_FIELDS_CHANNEL = 0;
  public static final int MAX_PLAYER_CHANNEL = 1;
  public static final int MIN_PLAYER_CHANNEL = 2;
  
  public static final double EMPTY_IMAGE_POINT = 0;
  public static final double OCCUPIED_IMAGE_POINT = 1;
  
  public static final INDArray ZEROS_PLAYGROUND_IMAGE = Nd4j.zeros(1, IMAGE_SIZE, IMAGE_SIZE);
  public static final INDArray ONES_PLAYGROUND_IMAGE = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
  
  public static final INDArray EMPTY_CONVOLUTIONAL_PLAYGROUND = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
  static {
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(EMPTY_FIELDS_CHANNEL, ONES_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(MAX_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(MIN_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
  }

  public static final int[] FIELD_1_2D = new int[] {0,0};
  public static final int[] FIELD_2_2D = new int[] {0,1};
  public static final int[] FIELD_3_2D = new int[] {0,2};
  public static final int[] FIELD_4_2D = new int[] {1,0};
  public static final int[] FIELD_5_2D = new int[] {1,1};
  public static final int[] FIELD_6_2D = new int[] {1,2};
  public static final int[] FIELD_7_2D = new int[] {2,0};
  public static final int[] FIELD_8_2D = new int[] {2,1};
  public static final int[] FIELD_9_2D = new int[] {2,2};

  /**
   * Non empty playground fields ("crosses or circles"), empty is zero.
   */
  public static final double MAX_PLAYER = 1.0;
  public static final double MIN_PLAYER = -1.0;
  public static final double EMPTY_FIELD_VALUE = 0.0;

  public static final INDArray CENTER_FIELD_MOVE = Nd4j.zeros(1, COLUMN_NUMBER).putScalar(FIELD_5, MAX_PLAYER);
  public static final INDArray LAST_CORNER_FIELD_MOVE = Nd4j.zeros(1, COLUMN_NUMBER).putScalar(FIELD_9, MAX_PLAYER);

  /**
   * Result labels interpretation. MAX_WIN - 9 > DRAW_VALUE > MIN_WIN + 9 for
   * MiniMax comparison. Any occupied playground field leads to a zero result.
   */
  public static final double MAX_WIN = 10;
  public static final double MIN_WIN = -10;
  public static final int DEPTH_ADVANTAGE = 1;
  public static final double DRAW_VALUE = 0;
  public static final double OCCUPIED = 0;

  public static final int SMALL_CAPACITY = 10;
}
