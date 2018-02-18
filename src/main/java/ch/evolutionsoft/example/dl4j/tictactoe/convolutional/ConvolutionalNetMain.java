package ch.evolutionsoft.example.dl4j.tictactoe.convolutional;

import static ch.evolutionsoft.example.dl4j.tictactoe.TicTacToeConstants.*;
import static ch.evolutionsoft.example.dl4j.tictactoe.commonnet.NeuralNetConstants.*;

import java.io.IOException;
import java.util.List;

import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.example.dl4j.tictactoe.commonnet.NeuralDataHelper;
import ch.evolutionsoft.example.dl4j.tictactoe.commonnet.TicTacToeNeuralDataConverter;

public class ConvolutionalNetMain {

  public static final double CONVOLUTION_LEARNING_RATE = 0.0097;

  public static final int CONVOLUTION_NUMBER_OF_EPOCHS = 100;

  public static final int NET_ITERATIONS = 2;

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
    Evaluation eval = new Evaluation(COLUMN_NUMBER);
    eval.eval(dataSet.getLabels(), output);

    if (logger.isInfoEnabled()) {
      logger.info(eval.stats());
    }

    INDArray graphSingleBatchInput1 = generateCenterFieldInputImages();
    INDArray graphSingleBatchInput2 = generateLastCornerFieldInputImages();

    logger.info("Input center field move: {}", graphSingleBatchInput1.getRow(0));
    INDArray centerFieldOpeningAnswer = graphNetwork.outputSingle(graphSingleBatchInput1);
    logger.info("Input last corner field move: {}", graphSingleBatchInput2.getRow(0));
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

    List<Pair<INDArray, INDArray>> allPlaygroundsResults = NeuralDataHelper.readAll();

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

  protected ComputationGraph buildNetwork() {

    String message = "Build model ...";
    logger.info(message);

    ComputationGraphConfiguration conf = createConvolutionalGraphConfiguration();
    ComputationGraph net = new ComputationGraph(conf);
    net.init();

    return net;
  }

  protected ComputationGraphConfiguration createConvolutionalGraphConfiguration() {

    return new NeuralNetConfiguration.Builder()
        .seed(DEFAULT_SEED)
        .iterations(NET_ITERATIONS)
        .updater(Updater.ADAM)
        .convolutionMode(ConvolutionMode.Strict)
        .learningRate(CONVOLUTION_LEARNING_RATE)
        .weightInit(WeightInit.XAVIER)
        .graphBuilder()
        .addInputs(DEFAULT_INPUT_LAYER_NAME)
        .addLayer("cnn0", new ConvolutionLayer.Builder()
            .kernelSize(1, 1)
            .stride(1, 1)
            .padding(0, 0)
            .nIn(IMAGE_CHANNELS)
            .nOut(CNN_OUTPUT_CHANNELS)
            .activation(Activation.ELU)
            .build(), DEFAULT_INPUT_LAYER_NAME)
        .addVertex("fc0-pre",
            new PreprocessorVertex(new CnnToFeedForwardPreProcessor(IMAGE_SIZE, IMAGE_SIZE, CNN_OUTPUT_CHANNELS)),
            "cnn0")
        .addLayer("fc0",
            new DenseLayer.Builder()
                .nIn(27)
                .nOut(131)
                .activation(Activation.SIGMOID)
                .build(),
            "fc0-pre")
        .addLayer(DEFAULT_OUTPUT_LAYER_NAME, new OutputLayer.Builder()
            .nIn(131)
            .nOut(9)
            .activation(Activation.SOFTMAX)
            .build(), "fc0")
        .setOutputs(DEFAULT_OUTPUT_LAYER_NAME)
        .pretrain(false)
        .backprop(true)
        .build();
  }

  protected EarlyStoppingConfiguration<ComputationGraph> createEarlyStoppingConfiguration(
      DataSetIterator dataSetIterator) {

    return new EarlyStoppingConfiguration.Builder<ComputationGraph>()
        .epochTerminationConditions(new MaxEpochsTerminationCondition(CONVOLUTION_NUMBER_OF_EPOCHS))
        .iterationTerminationConditions(new MaxScoreIterationTerminationCondition(DEFAULT_MAX_SCORE_EARLY_STOP))
        .scoreCalculator(new DataSetLossCalculatorCG(dataSetIterator, true))
        .evaluateEveryNEpochs(1)
        .modelSaver(new InMemoryModelSaver<>())
        .build();
  }
}
