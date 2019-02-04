package ch.evolutionsoft.example.dl4j.tictactoe.convolutional;

import static ch.evolutionsoft.net.game.NeuralNetConstants.DEFAULT_BATCH_SIZE;
import static ch.evolutionsoft.net.game.NeuralNetConstants.DEFAULT_FEATURE_EXAMPLE_NUMBER_LOG;
import static ch.evolutionsoft.net.game.NeuralNetConstants.DEFAULT_INPUT_LAYER_NAME;
import static ch.evolutionsoft.net.game.NeuralNetConstants.DEFAULT_MAX_SCORE_EARLY_STOP;
import static ch.evolutionsoft.net.game.NeuralNetConstants.DEFAULT_OUTPUT_LAYER_NAME;
import static ch.evolutionsoft.net.game.NeuralNetConstants.DEFAULT_SEED;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.IMAGE_CHANNELS;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.IMAGE_SIZE;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.MAX_PLAYER;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.OCCUPIED;

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
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.example.dl4j.tictactoe.feedforward.FeedForwardCommon;
import ch.evolutionsoft.net.game.NeuralDataHelper;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeNeuralDataConverter;

public class ConvolutionalMultiLabelNetMain {

  public static final double CONVOLUTION_LEARNING_RATE = 0.01;

  public static final int CONVOLUTION_NUMBER_OF_EPOCHS = 10;

  public static final int CNN_OUTPUT_CHANNELS = 3;

  private static final Logger logger = LoggerFactory.getLogger(ConvolutionalNetMain.class);

  public static void main(String[] args) throws Exception {

    ConvolutionalMultiLabelNetMain convolutionalNetMain = new ConvolutionalMultiLabelNetMain();

    ComputationGraph convolutionalNet = convolutionalNetMain.buildNetwork();

    if (logger.isInfoEnabled()) {
      logger.info(convolutionalNet.summary());
    }

    DataSet dataSet = convolutionalNetMain.trainNetwork(convolutionalNet);

    convolutionalNetMain.evaluateNetwork(convolutionalNet, dataSet);
  }

  protected void evaluateNetwork(ComputationGraph graphNetwork, DataSet dataSet) {

    INDArray output = graphNetwork.outputSingle(dataSet.getFeatures());
    EvaluationBinary eval = new EvaluationBinary(null);
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

    List<Pair<INDArray, INDArray>> allPlaygroundsResults = NeuralDataHelper.readAll(FeedForwardCommon.INPUTS_PATH, FeedForwardCommon.LABELS_PATH);

    List<Pair<INDArray, INDArray>> trainDataSetPairsList =
        TicTacToeNeuralDataConverter.generateMultiClassLabelsConvolutional(allPlaygroundsResults);

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
        .updater(new Adam(CONVOLUTION_LEARNING_RATE))
        .convolutionMode(ConvolutionMode.Strict);
  }

  ComputationGraphConfiguration createConvolutionalGraphConfiguration() {

    return new ComputationGraphConfiguration.GraphBuilder(createGeneralConfiguration())
        .addInputs(DEFAULT_INPUT_LAYER_NAME)
        .addLayer("cnn0", new ConvolutionLayer.Builder(1, 1)
            .stride(1, 1)
            .padding(0, 0)
            .nIn(IMAGE_CHANNELS)
            .nOut(22)
            .convolutionMode(ConvolutionMode.Same)
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .build(), DEFAULT_INPUT_LAYER_NAME)
        .addLayer("cnn1",
            new ConvolutionLayer.Builder(1, 1)
            .stride(1, 1)
            .padding(0, 0)
            .nIn(22)
            .nOut(22)
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .build(),
            "cnn0")
        .addLayer("cnn2",
            new ConvolutionLayer.Builder(1, 1)
            .stride(1, 1)
            .padding(0, 0)
            .nIn(22)
            .nOut(22)
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.RELU)
            .build(),
            "cnn1")
        .addVertex("fc1-pre",
            new PreprocessorVertex(new CnnToFeedForwardPreProcessor(IMAGE_SIZE, IMAGE_SIZE, 22)),
            "cnn2")
        .addLayer(DEFAULT_OUTPUT_LAYER_NAME, new OutputLayer.Builder(LossFunction.SQUARED_LOSS)
            .nIn(198)
            .nOut(9)
            .activation(Activation.SIGMOID)
            .weightInit(WeightInit.XAVIER)
            .build(), "fc1-pre")
        .setOutputs(DEFAULT_OUTPUT_LAYER_NAME)
        .build();
  }

  protected EarlyStoppingConfiguration<ComputationGraph> createEarlyStoppingConfiguration(
      DataSetIterator dataSetIterator) {

    return new EarlyStoppingConfiguration.Builder<ComputationGraph>()
        .epochTerminationConditions(new MaxEpochsTerminationCondition(CONVOLUTION_NUMBER_OF_EPOCHS))
        .iterationTerminationConditions(new MaxScoreIterationTerminationCondition(DEFAULT_MAX_SCORE_EARLY_STOP))
        .scoreCalculator(new DataSetLossCalculator(dataSetIterator, true))
        .evaluateEveryNEpochs(1)
        .modelSaver(new InMemoryModelSaver<>())
        .build();
  }
}
