package ch.evolutionsoft.example.dl4j.tictactoe.feedforward;

import static ch.evolutionsoft.net.game.NeuralNetConstants.*;

import java.io.IOException;
import java.net.URISyntaxException;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;

public class FeedForwardThreeLayerMain {

  private static final int FIRST_LAYER_NUMBER_OF_NODES = 50;
  private static final int SECOND_LAYER_NUMBER_OF_NODES = 50;

  private static final double LEARNING_RATE = .00025;

  private static final Logger logger = LoggerFactory.getLogger(FeedForwardThreeLayerMain.class);

  public static void main(String[] args) throws IOException, URISyntaxException, InterruptedException {

    FeedForwardThreeLayerMain hiddenLayerSetup = new FeedForwardThreeLayerMain();

    FeedForwardCommon feedForwardCommon = new FeedForwardCommon();
    MultiLayerNetwork net = feedForwardCommon.createNetworkModel(
        hiddenLayerSetup.createHiddenLayerConfiguration(hiddenLayerSetup.createGeneralConfiguration())
        .build());

    if (logger.isInfoEnabled()) {
      logger.info(net.summary());
    }

    feedForwardCommon.trainNetworkModel(net);

    feedForwardCommon.evaluateNetworkPerformance(net, feedForwardCommon.stackPlaygroundInputsLabels());
  }

  public NeuralNetConfiguration.Builder createGeneralConfiguration() {

    return new NeuralNetConfiguration.Builder()
        .seed(DEFAULT_SEED)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .weightInit(WeightInit.XAVIER)
        .updater(new Adam(LEARNING_RATE));
  }

  public NeuralNetConfiguration.ListBuilder createHiddenLayerConfiguration(
      NeuralNetConfiguration.Builder generalConfigBuilder) {

    return new NeuralNetConfiguration.ListBuilder(generalConfigBuilder)
        .layer(0, new DenseLayer.Builder()
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .nIn(TicTacToeConstants.COLUMN_COUNT)
            .nOut(FIRST_LAYER_NUMBER_OF_NODES)
            .name(DEFAULT_INPUT_LAYER_NAME)
            .build())
        .layer(1, new DenseLayer.Builder()
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .nIn(FIRST_LAYER_NUMBER_OF_NODES)
            .nOut(SECOND_LAYER_NUMBER_OF_NODES)
            .name(DEFAULT_HIDDEN_LAYER_NAME)
            .build())
        .layer(2, new OutputLayer.Builder()
            .activation(Activation.SOFTMAX)
            .nIn(SECOND_LAYER_NUMBER_OF_NODES)
            .nOut(TicTacToeConstants.COLUMN_COUNT)
            .name(DEFAULT_OUTPUT_LAYER_NAME)
            .build());
  }

}
