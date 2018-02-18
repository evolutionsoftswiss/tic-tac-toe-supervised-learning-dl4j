package ch.evolutionsoft.example.dl4j.tictactoe.commonnet;

import java.util.Random;

public final class NeuralNetConstants {

  private NeuralNetConstants() {
    // Hide constructor
  }

  public static final long DEFAULT_SEED = 2357;

  public static final Random randomGenerator = new Random(DEFAULT_SEED);

  public static final int DEFAULT_NUMBER_OF_EPOCHS = 50;

  public static final int DEFAULT_BATCH_SIZE = 10;

  public static final int DEFAULT_MAX_SCORE_EARLY_STOP = 100;

  public static final int DEFAULT_FEATURE_EXAMPLE_NUMBER_LOG = 1;

  public static final String DEFAULT_INPUT_LAYER_NAME = "InputLayer";
  public static final String DEFAULT_HIDDEN_LAYER_NAME = "HiddenLayer";
  public static final String DEFAULT_OUTPUT_LAYER_NAME = "OutputLayer";

}
