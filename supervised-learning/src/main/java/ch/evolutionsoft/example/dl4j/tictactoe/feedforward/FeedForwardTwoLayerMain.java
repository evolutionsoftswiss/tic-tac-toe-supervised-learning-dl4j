package ch.evolutionsoft.example.dl4j.tictactoe.feedforward;

import static ch.evolutionsoft.net.game.NeuralNetConstants.*;

import java.io.IOException;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.learning.config.RmsProp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;

public class FeedForwardTwoLayerMain {

  private static final int NUMBER_OF_NODES = 151;

  private static final Logger logger = LoggerFactory.getLogger(FeedForwardTwoLayerMain.class);

  private static final double LEARNING_RATE = 4e-5;

  public static void main(String[] args) throws IOException {

    FeedForwardTwoLayerMain twoLayerSetup = new FeedForwardTwoLayerMain();

    FeedForwardCommon feedForwardCommon = new FeedForwardCommon();
    MultiLayerNetwork net = feedForwardCommon.createNetworkModel(
        twoLayerSetup.createTwoLayerConfiguration(twoLayerSetup.createGeneralConfiguration()).build());

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
        .weightInit(WeightInit.DISTRIBUTION)
        .updater(new RmsProp(LEARNING_RATE, 0.98, 1e-8));
  }

  public NeuralNetConfiguration.ListBuilder createTwoLayerConfiguration(
      NeuralNetConfiguration.Builder generalConfigBuilder) {

    return new NeuralNetConfiguration.ListBuilder(generalConfigBuilder)
        .layer(0, new DenseLayer.Builder()
            .activation(new ActivationLReLU(0.1))
            .nIn(TicTacToeConstants.COLUMN_NUMBER)
            .nOut(NUMBER_OF_NODES)
            .build())
        .layer(1, new OutputLayer.Builder()
            .activation(Activation.SOFTMAX)
            .weightInit(new ConstantDistribution(0.05))
            .nIn(NUMBER_OF_NODES)
            .nOut(TicTacToeConstants.COLUMN_NUMBER)
            .build());
  }
}
