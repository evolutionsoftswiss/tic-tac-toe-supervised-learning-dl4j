package ch.evolutionsoft.example.dl4j.tictactoe.convolutional;

import static ch.evolutionsoft.net.game.NeuralNetConstants.*;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;

import java.io.IOException;
import java.util.List;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.example.dl4j.tictactoe.feedforward.FeedForwardCommon;
import ch.evolutionsoft.net.game.NeuralDataHelper;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeNeuralDataConverter;

public class ConvolutionalNetMain {

  public static final double LEARNING_RATE = 0.031;

  public static final int NUMBER_OF_EPOCHS = 500;

  public static final int CNN_OUTPUT_CHANNELS = 3;

  private static final Logger logger = LoggerFactory.getLogger(ConvolutionalNetMain.class);

  public static void main(String[] args) throws Exception {

    ConvolutionalNetMain convolutionalNetMain = new ConvolutionalNetMain();

    ComputationGraph convolutionalNet = convolutionalNetMain.buildNetwork();

    if (logger.isInfoEnabled()) {
      logger.info(convolutionalNet.summary());
    }

    DataSet dataSet = convolutionalNetMain.trainNetwork(convolutionalNet);

    convolutionalNetMain.evaluateNetwork(convolutionalNet, dataSet);
  }

  protected void evaluateNetwork(ComputationGraph graphNetwork, DataSet dataSet) {

    INDArray output = graphNetwork.outputSingle(dataSet.getFeatures());
    Evaluation eval = new Evaluation(COLUMN_COUNT);
    eval.eval(dataSet.getLabels(), output);

    if (logger.isInfoEnabled()) {
      logger.info(eval.stats());
    }

    INDArray graphSingleBatchInput1 = generateCenterFieldInputImages();
    INDArray graphSingleBatchInput2 = generateLastCornerFieldInputImages();

    logger.info("Input center field move: {}", graphSingleBatchInput1);
    INDArray centerFieldOpeningAnswer = graphNetwork.outputSingle(graphSingleBatchInput1);
    logger.info("Input last corner field move: {}", graphSingleBatchInput2);
    INDArray cornerFieldOpeningAnswer = graphNetwork.outputSingle(graphSingleBatchInput2);

    logger.info("Answer to center field opening: {}", centerFieldOpeningAnswer);
    logger.info("Answer to last corner field opening: {}", cornerFieldOpeningAnswer);
  }

  protected INDArray generateCenterFieldInputImages() {

    INDArray middleFieldMove = Nd4j.zeros(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
    emptyImage1.putScalar(0, 1, 1, OCCUPIED);
    middleFieldMove.putRow(0, emptyImage1);
    middleFieldMove.putScalar(1, 1, 1, MAX_PLAYER);
    INDArray graphSingleBatchInput1 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput1.putRow(0, middleFieldMove);
    return graphSingleBatchInput1;
  }

  protected INDArray generateLastCornerFieldInputImages() {

    INDArray cornerFieldMove = Nd4j.zeros(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    INDArray emptyImage2 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
    emptyImage2.putScalar(0, 2, 2, OCCUPIED);
    cornerFieldMove.putRow(0, emptyImage2);
    cornerFieldMove.putScalar(1, 2, 2, MAX_PLAYER);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, cornerFieldMove);
    return graphSingleBatchInput2;
  }

  protected DataSet trainNetwork(ComputationGraph net)
      throws IOException {

    List<Pair<INDArray, INDArray>> allPlaygroundsResults =
        NeuralDataHelper.readAll(FeedForwardCommon.INPUTS_PATH, FeedForwardCommon.LABELS_PATH);

    List<Pair<INDArray, INDArray>> trainDataSetPairsList =
        TicTacToeNeuralDataConverter.convertMiniMaxPlaygroundLabelsToConvolutionalData(allPlaygroundsResults);

    NeuralDataHelper.printRandomConvolutionalNetInputAndLabels(trainDataSetPairsList,
        DEFAULT_FEATURE_EXAMPLE_NUMBER_LOG);

    DataSetIterator dataSetIterator = new INDArrayDataSetIterator(trainDataSetPairsList, DEFAULT_BATCH_SIZE);

    EarlyStoppingConfiguration<ComputationGraph> earlyStoppingConfiguration =
        createEarlyStoppingConfiguration(dataSetIterator);

    EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(earlyStoppingConfiguration, net, dataSetIterator);

    trainer.fit();

    Pair<INDArray, INDArray> stackedPlaygroundLabels =
        TicTacToeNeuralDataConverter.stackConvolutionalPlaygroundLabels(trainDataSetPairsList);

    return new org.nd4j.linalg.dataset.DataSet(stackedPlaygroundLabels.getFirst(), stackedPlaygroundLabels.getSecond());
  }

  public ComputationGraph buildNetwork() {

    String message = "Build model ...";
    logger.info(message);

    ComputationGraphConfiguration conf = createConvolutionalGraphConfiguration();
    ComputationGraph net = new ComputationGraph(conf);
    net.init();

    return net;
  }

  NeuralNetConfiguration.Builder createGeneralConfiguration() {

    return new NeuralNetConfiguration.Builder()
        .seed(DEFAULT_SEED)
        .updater(new Adam(LEARNING_RATE))
        .convolutionMode(ConvolutionMode.Strict)
        .weightInit(WeightInit.RELU);
  }

  ComputationGraphConfiguration createConvolutionalGraphConfiguration() {

    return new ComputationGraphConfiguration.GraphBuilder(createGeneralConfiguration())
        .addInputs("input").setInputTypes(InputType.convolutional(3, 3, 3))
        // block1
        .addLayer("block1_conv1",
            new ConvolutionLayer.Builder(2, 2).stride(1, 1).nIn(3).nOut(7).hasBias(false)
                .build(),
            "input")
        .addLayer("block1_conv1_bn", new BatchNormalization(), "block1_conv1")
        .addLayer("block1_conv1_act", new ActivationLayer(Activation.RELU), "block1_conv1_bn")
        .addLayer("block1_conv2",
            new ConvolutionLayer.Builder(2, 2).stride(1, 1).padding(1, 1).nOut(14).hasBias(false)
                .build(),
            "block1_conv1_act")
        .addLayer("block1_conv2_bn", new BatchNormalization(), "block1_conv2")
        .addLayer("block1_conv2_act", new ActivationLayer(Activation.RELU), "block1_conv2_bn")

        // residual1
        .addLayer("residual1_conv",
            new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(14).hasBias(false)
                .convolutionMode(ConvolutionMode.Same).build(),
            "block1_conv2_act")
        .addLayer("residual1", new BatchNormalization(), "residual1_conv")

        // block2
        .addLayer("block2_sepconv1",
            new SeparableConvolution2D.Builder(2, 2).nOut(14).hasBias(false).convolutionMode(ConvolutionMode.Same)
                .build(),
            "block1_conv2_act")
        .addLayer("block2_sepconv1_bn", new BatchNormalization(), "block2_sepconv1")
        .addLayer("block2_sepconv1_act", new ActivationLayer(Activation.RELU), "block2_sepconv1_bn")
        .addLayer("block2_sepconv2",
            new SeparableConvolution2D.Builder(2, 2).nOut(14).hasBias(false).convolutionMode(ConvolutionMode.Same)
                .build(),
            "block2_sepconv1_act")
        .addLayer("block2_sepconv2_bn", new BatchNormalization(), "block2_sepconv2")
        .addLayer("block2_pool",
            new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same).build(),
            "block2_sepconv2_bn")
        
        .addVertex("add1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block2_pool", "residual1")
        
        .addLayer(DEFAULT_OUTPUT_LAYER_NAME, new OutputLayer.Builder()
            .nOut(9)
            .activation(Activation.SOFTMAX)
            .build(), "add1")
        .setOutputs(DEFAULT_OUTPUT_LAYER_NAME)
        .build();
  }

  protected EarlyStoppingConfiguration<ComputationGraph> createEarlyStoppingConfiguration(
      DataSetIterator dataSetIterator) {

    return new EarlyStoppingConfiguration.Builder<ComputationGraph>()
        .epochTerminationConditions(new MaxEpochsTerminationCondition(NUMBER_OF_EPOCHS))
        .iterationTerminationConditions(new MaxScoreIterationTerminationCondition(DEFAULT_MAX_SCORE_EARLY_STOP))
        .scoreCalculator(new DataSetLossCalculator(dataSetIterator, true))
        .evaluateEveryNEpochs(1)
        .modelSaver(new InMemoryModelSaver<>())
        .build();
  }
}
