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

/**
 * 
 * 
 * File storedNetworkFile = new File("storedNetworkFile.zip");
 * ModelSerializer.writeModel(net, storedNetworkFile, false);
 */
public class FeedForwardHiddenLayerMain {

  private static final double LEARNING_RATE = 0.83;

  private static final Logger logger = LoggerFactory.getLogger(FeedForwardHiddenLayerMain.class);

  public static void main(String[] args) throws IOException {

    FeedForwardHiddenLayerMain hiddenLayerSetup = new FeedForwardHiddenLayerMain();

    FeedForwardCommon feedForwardCommon = new FeedForwardCommon();
    MultiLayerNetwork net = feedForwardCommon.createNetworkModel(hiddenLayerSetup.createHiddenLayerConfiguration());

    if (logger.isInfoEnabled()) {
      logger.info(net.summary());
    }

    DataSet dataSet = feedForwardCommon.trainNetworkModel(net);

    feedForwardCommon.evaluateNetworkPerformance(net, dataSet);
  }

  protected MultiLayerConfiguration createHiddenLayerConfiguration() {

    return new NeuralNetConfiguration.Builder()
        .seed(DEFAULT_SEED)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .iterations(NET_ITERATIONS)
        .learningRate(LEARNING_RATE)
        .weightInit(WeightInit.XAVIER)
        .updater(new Nesterovs(NESTEROVS_MOMENTUM))
        .list()
        .layer(0, new DenseLayer.Builder()
            .activation(Activation.SOFTSIGN)
            .nIn(9)
            .nOut(75)
            .name(DEFAULT_INPUT_LAYER_NAME)
            .build())
        .layer(1, new DenseLayer.Builder()
            .activation(Activation.SIGMOID)
            .nIn(75)
            .nOut(35)
            .name(DEFAULT_HIDDEN_LAYER_NAME)
            .build())
        .layer(2, new OutputLayer.Builder()
            .activation(Activation.SOFTMAX)
            .nIn(35)
            .nOut(9)
            .name(DEFAULT_OUTPUT_LAYER_NAME)
            .build())
        .pretrain(false)
        .backprop(true)
        .build();
  }

}
