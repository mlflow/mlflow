package org.mlflow.sagemaker;

import com.fasterxml.jackson.core.JsonProcessingException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import ml.combust.mleap.runtime.MleapContext;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import ml.combust.mleap.runtime.frame.Row;
import ml.combust.mleap.runtime.frame.Transformer;
import ml.combust.mleap.runtime.javadsl.BundleBuilder;
import ml.combust.mleap.runtime.javadsl.ContextBuilder;
import org.mlflow.utils.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.collection.JavaConverters;
import scala.collection.Seq;

/** A {@link org.mlflow.sagemaker.Predictor} implementation for the MLeap model flavor */
public class MLeapPredictor extends Predictor {
  private final Transformer pipelineTransformer;
  private final LeapFrameSchema inputSchema;

  private static final Seq<String> predictionColumnNames;
  private static final Logger logger = LoggerFactory.getLogger(MLeapPredictor.class);

  static {
    List<String> predictionColumnList = Arrays.asList("prediction");
    predictionColumnNames =
        JavaConverters.asScalaIteratorConverter(predictionColumnList.iterator()).asScala().toSeq();
  }

  /**
   * Constructs an {@link MLeapPredictor}
   *
   * @param modelDataPath The path to the serialized MLeap model
   * @param inputSchema The path to JSON-formatted file containing the input schema that the model
   *     accepts
   */
  public MLeapPredictor(String modelDataPath, String inputSchemaPath) {
    MleapContext mleapContext = new ContextBuilder().createMleapContext();
    BundleBuilder bundleBuilder = new BundleBuilder();
    this.pipelineTransformer = bundleBuilder.load(new File(modelDataPath), mleapContext).root();
    try {
      this.inputSchema = LeapFrameSchema.fromPath(inputSchemaPath);
    } catch (IOException e) {
      logger.error(e);
      throw new PredictorLoadingException(String.format(
          "Failed to load model input schema from specified path: %s", inputSchemaPath));
    }
  }

  @Override
  protected PredictorDataWrapper predict(PredictorDataWrapper input)
      throws PredictorEvaluationException {
    PandasRecordOrientedDataFrame pandasFrame = null;
    try {
      pandasFrame = PandasRecordOrientedDataFrame.fromJson(input.toJson());
    } catch (IOException e) {
      logger.error(
          "Encountered a JSON conversion error during conversion of Pandas dataframe to LeapFrame.",
          e);
      throw new PredictorEvaluationException(
          "Failed to transform input into a JSON representation of an MLeap dataframe."
          + "Please ensure that the input is a JSON-serialized Pandas Dataframe"
          + "with the `record` orientation");
    }

    DefaultLeapFrame leapFrame = null;
    try {
      leapFrame = pandasFrame.toLeapFrame(this.inputSchema);
    } catch (MissingSchemaFieldException e) {
      throw new PredictorEvaluationException(
          String.format("The input dataframe is missing the following required field: %s",
              e.getMissingFieldName()));
    } catch (Exception e) {
      logger.error(
          "Encountered an unknown error during conversion of Pandas dataframe to LeapFrame.", e);
      throw new PredictorEvaluationException(
          "An unknown error occurred while converting the input dataframe to a LeapFrame.");
    }

    DefaultLeapFrame predictionsFrame =
        this.pipelineTransformer.transform(leapFrame).get().select(predictionColumnNames).get();
    Seq<Row> predictionRows = predictionsFrame.collect();
    Iterable<Row> predictionRowsIterable =
        JavaConverters.asJavaIterableConverter(predictionRows).asJava();
    List<Object> predictions = new ArrayList<Object>();
    for (Row row : predictionRowsIterable) {
      predictions.add(row.getRaw(0));
    }

    String predictionsJson = null;
    try {
      predictionsJson = SerializationUtils.toJson(predictions);
    } catch (JsonProcessingException e) {
      logger.error("Encountered an error while serializing the output dataframe.", e);
      throw new PredictorEvaluationException(
          "Failed to serialize prediction results as a JSON list!");
    }
    return new PredictorDataWrapper(predictionsJson, PredictorDataWrapper.ContentType.Json);
  }

  /** @return The underlying MLeap pipeline transformer that this predictor uses for inference */
  public Transformer getPipeline() {
    return this.pipelineTransformer;
  }
}
