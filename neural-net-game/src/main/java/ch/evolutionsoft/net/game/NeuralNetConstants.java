package ch.evolutionsoft.net.game;

import java.util.Random;

public final class NeuralNetConstants {

  private NeuralNetConstants() {
    // Hide constructor
  }

  public static final long DEFAULT_SEED = 2357;

  public static final Random randomGenerator = new Random(DEFAULT_SEED);

  public static final int DEFAULT_NUMBER_OF_EPOCHS = 10000;

  public static final int DEFAULT_BATCH_SIZE = 4608;

  public static final int DEFAULT_MAX_SCORE_EARLY_STOP = 10;
  
  public static final int MEDIUM_CAPACITY = 5000;

  public static final int DEFAULT_FEATURE_EXAMPLE_NUMBER_LOG = 1;

  public static final String DEFAULT_INPUT_LAYER_NAME = "InputLayer";
  public static final String DEFAULT_HIDDEN_LAYER_NAME = "HiddenLayer";
  public static final String DEFAULT_OUTPUT_LAYER_NAME = "OutputLayer";
  
  public static final double ZERO = 0.0;

  public static final double DOUBLE_COMPARISON_EPSILON = 0.01;

  public static final int ONE = 1;
  public static final int TWO = 2;
  public static final int THREE = 3;
  public static final int FOUR = 4;
  public static final int FIVE = 5;
  
}
