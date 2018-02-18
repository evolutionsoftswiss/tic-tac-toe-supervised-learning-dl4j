package ch.evolutionsoft.example.dl4j.tictactoe.feedforward;

import static ch.evolutionsoft.example.dl4j.tictactoe.commonnet.NeuralNetConstants.*;
import static ch.evolutionsoft.example.dl4j.tictactoe.feedforward.FeedForwardCommon.*;

import java.io.IOException;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FeedForwardTwoLayerMain {

  private static final Logger logger = LoggerFactory.getLogger(FeedForwardTwoLayerMain.class);

  public static void main(String[] args) throws IOException {

    FeedForwardTwoLayerMain twoLayerSetup = new FeedForwardTwoLayerMain();

    FeedForwardCommon feedForwardCommon = new FeedForwardCommon();
    MultiLayerNetwork net = feedForwardCommon.createNetworkModel(twoLayerSetup.createTwoLayerConfiguration());

    if (logger.isInfoEnabled()) {
      logger.info(net.summary());
    }

    DataSet dataSet = feedForwardCommon.trainNetworkModel(net);

    feedForwardCommon.evaluateNetworkPerformance(net, dataSet);
  }

  protected MultiLayerConfiguration createTwoLayerConfiguration() {

    return new NeuralNetConfiguration.Builder()
        .seed(DEFAULT_SEED)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .iterations(NET_ITERATIONS)
        .learningRate(0.99)
        .weightInit(WeightInit.XAVIER)
        .updater(new Nesterovs(NESTEROVS_MOMENTUM))
        .list()
        .layer(0, new DenseLayer.Builder()
            .activation(Activation.SOFTSIGN)
            .nIn(9)
            .nOut(161)
            .build())
        .layer(1, new OutputLayer.Builder()
            .activation(Activation.SOFTMAX)
            .nIn(161)
            .nOut(9)
            .build())
        .pretrain(false)
        .backprop(true)
        .build();
  }
}
