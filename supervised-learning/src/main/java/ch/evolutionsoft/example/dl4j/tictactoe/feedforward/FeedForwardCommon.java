package ch.evolutionsoft.example.dl4j.tictactoe.feedforward;

import static ch.evolutionsoft.net.game.NeuralNetConstants.*;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;

import java.io.IOException;
import java.util.List;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.NeuralDataHelper;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeNeuralDataConverter;

public class FeedForwardCommon {

  private static final Logger logger = LoggerFactory.getLogger(FeedForwardCommon.class);

  private static final int NUMBER_OF_EPOCHS = 2000;

  public static String INPUTS_PATH = "/inputs.txt";
  public static String LABELS_PATH = "/labels.txt";

  public MultiLayerNetwork createNetworkModel(MultiLayerConfiguration multiLayerConfiguration) {

    String message = "Build model ...";
    logger.info(message);

    MultiLayerNetwork net = new MultiLayerNetwork(multiLayerConfiguration);
    net.init();

    return net;
  }

  public void trainNetworkModel(MultiLayerNetwork net) throws IOException {

    String message = "Generate adapted net input and labels ...";
    logger.info(message);

    List<Pair<INDArray, INDArray>> allPlaygrounds = NeuralDataHelper.readAll(INPUTS_PATH, LABELS_PATH);
    List<Pair<INDArray, INDArray>> convertedMiniMaxLabels = TicTacToeNeuralDataConverter.convertMiniMaxLabels(allPlaygrounds);

    NeuralDataHelper.printRandomMiniMaxData(allPlaygrounds, DEFAULT_FEATURE_EXAMPLE_NUMBER_LOG);
    NeuralDataHelper.printRandomFeedForwardNetInputAndLabels(
        convertedMiniMaxLabels, DEFAULT_FEATURE_EXAMPLE_NUMBER_LOG);
    
    net.addListeners(new ScoreIterationListener(DEFAULT_NUMBER_OF_PRINT_EPOCHS));
    
    for (int epochNumber = 0; epochNumber < NUMBER_OF_EPOCHS; epochNumber++) {
      
      DataSet randomBalancedDataSet = getTrainDataSetWithMaxLabelExampleSize(convertedMiniMaxLabels);
      /*DataSet dataSet = new org.nd4j.linalg.dataset.DataSet(
          stackedPlaygrounds.getFirst(),
          stackedPlaygrounds.getSecond()
          );
      */
      net.fit(randomBalancedDataSet);
    }
  }

  public DataSet getTrainDataSetWithMaxLabelExampleSize(List<Pair<INDArray, INDArray>> convertedMiniMaxLabels) {

    Pair<INDArray, INDArray> stackedPlaygroundLabels =
        TicTacToeNeuralDataConverter.stackFeedForwardPlaygroundLabels(convertedMiniMaxLabels);
    
    //labelStatistics1 = new int[] {1449, 421, 581, 313, 618, 227, 360, 170, 318}
    INDArray labelStatisticsNdArray = stackedPlaygroundLabels.getSecond().sum(0);
    
    double[] labelStatistics = labelStatisticsNdArray.toDoubleVector();
    double minOccurance = Nd4j.min(labelStatisticsNdArray).toDoubleVector()[0];
    
    int stackedSize = (int) minOccurance * COLUMN_COUNT;
    
    INDArray stackedPlaygrounds = Nd4j.zeros(stackedSize, COLUMN_COUNT);
    INDArray stackedLabels = Nd4j.zeros(stackedSize, COLUMN_COUNT);

    double[] ratios = new double[] {minOccurance / labelStatistics[0], minOccurance / labelStatistics[1],minOccurance / labelStatistics[2],
        minOccurance / labelStatistics[3],minOccurance / labelStatistics[4],minOccurance / labelStatistics[5],
        minOccurance / labelStatistics[6],minOccurance / labelStatistics[7],minOccurance / labelStatistics[8]};
    
    for (int totalIndex = 0, stackedIndex = 0; stackedIndex < stackedSize && totalIndex < 4520; totalIndex++) {

      INDArray currentLabel = stackedPlaygroundLabels.getSecond().getRow(totalIndex);
      int currentLabelIndex = Nd4j.getExecutioner().execAndReturn(new IMax(currentLabel)).getFinalResult().intValue();
      
      if (randomGenerator.nextDouble() <= ratios[currentLabelIndex]) {
      
        INDArray currentPlayground = stackedPlaygroundLabels.getFirst().getRow(totalIndex);
        stackedPlaygrounds.putRow(stackedIndex, currentPlayground);
  
        stackedLabels.putRow(stackedIndex, currentLabel);
        
        stackedIndex++;
      }
    }

    return new org.nd4j.linalg.dataset.DataSet(stackedPlaygrounds, stackedLabels);
  }

  public DataSet stackPlaygroundInputsLabels() {

    List<Pair<INDArray, INDArray>> allPlaygrounds = NeuralDataHelper.readAll(INPUTS_PATH, LABELS_PATH);
    List<Pair<INDArray, INDArray>> convertedMiniMaxLabels = TicTacToeNeuralDataConverter.convertMiniMaxLabels(allPlaygrounds);

    Pair<INDArray, INDArray> stackedPlaygroundLabels =
        TicTacToeNeuralDataConverter.stackFeedForwardPlaygroundLabels(convertedMiniMaxLabels);

    return new org.nd4j.linalg.dataset.DataSet(stackedPlaygroundLabels.getFirst(), stackedPlaygroundLabels.getSecond());
  }

  public void evaluateNetworkPerformance(MultiLayerNetwork net, DataSet dataSet) {

    INDArray output = net.output(dataSet.getFeatures());
    Evaluation eval = new Evaluation(COLUMN_COUNT);
    eval.eval(dataSet.getLabels(), output);

    if (logger.isInfoEnabled()) {
      logger.info(eval.stats());
    }

    logger.info("Answer to center field opening: {}", net.output(CENTER_FIELD_MOVE));
    logger.info("Answer to last corner field opening: {}", net.output(LAST_CORNER_FIELD_MOVE));
  }

  protected EarlyStoppingConfiguration<MultiLayerNetwork> createEarlyStoppingConfiguration(
      DataSetIterator dataSetIterator) {

    return new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
        .epochTerminationConditions(new MaxEpochsTerminationCondition(NUMBER_OF_EPOCHS))
        .iterationTerminationConditions(new MaxScoreIterationTerminationCondition(DEFAULT_MAX_SCORE_EARLY_STOP))
        .scoreCalculator(new DataSetLossCalculator(dataSetIterator, true))
        .evaluateEveryNEpochs(50)
        .modelSaver(new InMemoryModelSaver<>())
        .build();
  }
}
