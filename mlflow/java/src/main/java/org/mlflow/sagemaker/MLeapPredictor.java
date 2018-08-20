package org.mlflow.sagemaker;

import org.mlflow.mleap.LeapFrameUtils;
import org.mlflow.mleap.LeapFrameSchema;
import org.mlflow.utils.SerializationUtils;

import ml.combust.mleap.runtime.MleapContext;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import ml.combust.mleap.runtime.frame.Transformer;
import ml.combust.mleap.runtime.frame.Row;
import ml.combust.mleap.runtime.javadsl.BundleBuilder;
import ml.combust.mleap.runtime.javadsl.ContextBuilder;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import scala.collection.JavaConverters;
import scala.collection.Seq;

import com.fasterxml.jackson.core.JsonProcessingException;

/** A {@link org.mlflow.sagemaker.Predictor} implementation for the MLeap model flavor */
public class MLeapPredictor extends Predictor {
  private final Transformer pipelineTransformer;
  private final LeapFrameSchema inputSchema;

  private static final Seq<String> predictionColumnNames;

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
   * accepts
   */
  public MLeapPredictor(String modelDataPath, String inputSchemaPath) {
    MleapContext mleapContext = new ContextBuilder().createMleapContext();
    BundleBuilder bundleBuilder = new BundleBuilder();
    this.pipelineTransformer = bundleBuilder.load(new File(modelDataPath), mleapContext).root();
    try {
      this.inputSchema = LeapFrameSchema.fromPath(inputSchemaPath);
    } catch (IOException e) {
      e.printStackTrace();
      throw new PredictorLoadingException(String.format(
          "Failed to load model input schema from specified path: %s", inputSchemaPath));
    }
  }

  @Override
  protected DataFrame predict(DataFrame input) throws PredictorEvaluationException {
    Optional<String> leapFrameJson = Optional.empty();
    try {
      leapFrameJson = Optional.of(inputSchema.applyToPandasRecordJson(input.toJson()));
    } catch (IOException e) {
      e.printStackTrace();
      throw new PredictorEvaluationException(
          "Failed to transform input into a JSON representation of an MLeap dataframe."
          + "Please ensure that the input is a JSON-serialized Pandas Dataframe"
          + "with the `record` orientation");
    }
    DefaultLeapFrame leapFrame = LeapFrameUtils.getLeapFrameFromJson(leapFrameJson.get());

    DefaultLeapFrame predictionsFrame =
        this.pipelineTransformer.transform(leapFrame).get().select(predictionColumnNames).get();
    Seq<Row> predictionRows = predictionsFrame.collect();
    Iterable<Row> predictionRowsIterable =
        JavaConverters.asJavaIterableConverter(predictionRows).asJava();
    List<Object> predictions = new ArrayList<Object>();
    for (Row row : predictionRowsIterable) {
      predictions.add(row.getRaw(0));
    }

    Optional<String> predictionsJson = Optional.empty();
    try {
      predictionsJson = Optional.of(SerializationUtils.toJson(predictions));
    } catch (JsonProcessingException e) {
      e.printStackTrace();
      throw new PredictorEvaluationException(
          "Failed to serialize prediction results as a JSON list!");
    }
    return DataFrame.fromJson(predictionsJson.get());
  }

  /** @return The underlying MLeap pipeline transformer that this predictor uses for inference */
  public Transformer getPipeline() {
    return this.pipelineTransformer;
  }
}
