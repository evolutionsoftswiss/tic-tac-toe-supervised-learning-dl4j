package ch.evolutionsoft.example.dl4j.tictactoe.commonnet;

import static ch.evolutionsoft.example.dl4j.tictactoe.TicTacToeConstants.*;
import static ch.evolutionsoft.example.dl4j.tictactoe.commonnet.NeuralNetConstants.*;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NeuralDataHelper {

  public static final String IND_ARRAY_VALUE_SEPARATOR = ":";
  public static final String INPUT = "Example Neural Net Input";
  public static final String LABEL = " Label=";
  private static final char NEW_LINE = '\n';

  private static final Logger logger = LoggerFactory.getLogger(NeuralDataHelper.class);

  private NeuralDataHelper() {
    // Hide constructor
  }

  public static List<Pair<INDArray, INDArray>> printRandomMiniMaxData(
      List<Pair<INDArray, INDArray>> allPlaygroundsResults,
      int numberOfExamples) {

    int numberOfPlaygroundsAndLabels = allPlaygroundsResults.size();
    List<Pair<INDArray, INDArray>> chosenRandomPairs = new LinkedList<>();

    for (int n = 0; n < numberOfExamples && n < numberOfPlaygroundsAndLabels; n++) {

      Pair<INDArray, INDArray> currentRandomPair =
          allPlaygroundsResults.get(randomGenerator.nextInt(numberOfPlaygroundsAndLabels));
      logger.info("MiniMax Input and Output: {}", currentRandomPair);

      chosenRandomPairs.add(currentRandomPair);
    }

    return chosenRandomPairs;
  }

  public static List<Pair<INDArray,INDArray>> printRandomFeedForwardNetInputAndLabels(List<Pair<INDArray, INDArray>> playgroundsAndAdaptedLabels,
      int numberOfExamples) {

    int numberOfInputRows = playgroundsAndAdaptedLabels.size();

    List<Pair<INDArray,INDArray>> chosenExamples = new LinkedList<>();
    for (int n = 0; n < numberOfExamples && n < numberOfInputRows; n++) {

      int randomRow = randomGenerator.nextInt(numberOfInputRows);

      INDArray currentInput = playgroundsAndAdaptedLabels.get(randomRow).getFirst();
      INDArray currentLabel = playgroundsAndAdaptedLabels.get(randomRow).getSecond();
      
      chosenExamples.add(new Pair<>(currentInput, currentLabel));
      
      logger.info(INPUT + currentInput + LABEL + currentLabel);
    }
    
    return chosenExamples;
  }

  public static void printRandomConvolutionalNetInputAndLabels(List<Pair<INDArray, INDArray>> adaptedFetauresLabels,
      int numberOfExamples) {

    int numberOfInputsLables = adaptedFetauresLabels.size();

    for (int n = 0; n < numberOfExamples && n < numberOfInputsLables; n++) {

      int randomRow = randomGenerator.nextInt(numberOfInputsLables);

      logger.info(INPUT + NEW_LINE + adaptedFetauresLabels.get(randomRow).getFirst() + NEW_LINE + LABEL +
                  adaptedFetauresLabels.get(randomRow).getSecond());
    }
  }

  public static List<Pair<INDArray, INDArray>> readAll() {

    List<Pair<INDArray, INDArray>> allPlaygroundsResult = new LinkedList<>();

    INDArray inputs = readInputs();
    INDArray labels = readLabels();

    for (int row = 0; row < inputs.shape()[0]; row++) {

      allPlaygroundsResult.add(new Pair<INDArray, INDArray>(inputs.getRow(row), labels.getRow(row)));
    }

    return allPlaygroundsResult;
  }

  public static INDArray readInputs() {

    return Nd4j.readTxtString(NeuralDataHelper.class.getResourceAsStream("/inputs.txt"), IND_ARRAY_VALUE_SEPARATOR);
  }

  public static INDArray readLabels() {

    return Nd4j.readTxtString(NeuralDataHelper.class.getResourceAsStream("/labels.txt"), IND_ARRAY_VALUE_SEPARATOR);
  }

  public static void writeData(List<Pair<INDArray, INDArray>> allPlaygroundsResults) {

    Pair<INDArray, INDArray> stackedPlaygroundsLabels =
        TicTacToeNeuralDataConverter.stackFeedForwardPlaygroundLabels(allPlaygroundsResults);

    Nd4j.writeTxt(stackedPlaygroundsLabels.getFirst(), "inputs.txt");
    Nd4j.writeTxt(stackedPlaygroundsLabels.getSecond(), "labels.txt");

  }

}
