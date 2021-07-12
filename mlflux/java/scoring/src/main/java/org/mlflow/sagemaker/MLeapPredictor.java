package org.mlflow.sagemaker;

import com.fasterxml.jackson.core.JsonProcessingException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import ml.combust.mleap.core.types.StructType;
import ml.combust.mleap.runtime.MleapContext;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import ml.combust.mleap.runtime.frame.Transformer;
import ml.combust.mleap.runtime.javadsl.BundleBuilder;
import ml.combust.mleap.runtime.javadsl.ContextBuilder;
import ml.combust.mleap.runtime.javadsl.LeapFrameSupport;
import org.mlflow.utils.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A {@link org.mlflow.sagemaker.Predictor} implementation for the MLeap model flavor */
public class MLeapPredictor implements Predictor {
  private final Transformer pipelineTransformer;

  // As in the `pyfunc` wrapper for Spark models, we expect output dataframes
  // to have a `prediction` column that contains model predictions. Only entries in this
  // column are returned in the response to a query.`pyfunc` reference:
  // https://github.com/mlflow/mlflow/blob/f4869beec5cd2220d1bf01861d80f7145a8601bf/mlflow/
  // spark.py#L248
  private static final String PREDICTION_COLUMN_NAME = "prediction";
  private static final Logger logger = LoggerFactory.getLogger(MLeapPredictor.class);
  private final LeapFrameSupport leapFrameSupport;
  private final StructType inputSchema;

  /**
   * Constructs an {@link MLeapPredictor}
   *
   * @deprecated Use {@link #MLeapPredictor(String)} instead.
   * @param modelDataPath The path to the serialized MLeap model
   * @param inputSchemaPath The path to the serialized MLeap schema
   */
  @java.lang.Deprecated
  public MLeapPredictor(String modelDataPath, String inputSchemaPath) {
    MleapContext mleapContext = new ContextBuilder().createMleapContext();
    BundleBuilder bundleBuilder = new BundleBuilder();
    this.leapFrameSupport = new LeapFrameSupport();

    this.pipelineTransformer = bundleBuilder.load(new File(modelDataPath), mleapContext).root();
    try {
      this.inputSchema = new MLeapSchemaReader().fromFile(inputSchemaPath);
    } catch (Exception e) {
      logger.error("Could not read the model input schema from the specified path", e);
      throw new PredictorLoadingException(
              String.format(
                "Failed to load model input schema from specified path: %s", inputSchemaPath));
    }
  }

  /**
   * Constructs an {@link MLeapPredictor}
   *
   * @param modelDataPath The path to the serialized MLeap model
   */
  public MLeapPredictor(String modelDataPath) {
    MleapContext mleapContext = new ContextBuilder().createMleapContext();
    BundleBuilder bundleBuilder = new BundleBuilder();
    this.leapFrameSupport = new LeapFrameSupport();

    this.pipelineTransformer = bundleBuilder.load(new File(modelDataPath), mleapContext).root();
    this.inputSchema = this.pipelineTransformer.inputSchema();
  }

  @Override
  public PredictorDataWrapper predict(PredictorDataWrapper input)
      throws PredictorEvaluationException {
    PandasSplitOrientedDataFrame pandasFrame;
    try {
      pandasFrame = PandasSplitOrientedDataFrame.fromJson(input.toJson());
    } catch (IOException e) {
      logger.error(
          "Encountered a JSON parsing error during conversion of input to a Pandas DataFrame"
              + " representation.",
          e);
      throw new PredictorEvaluationException(
          "Encountered a JSON parsing error while transforming input into a Pandas DataFrame"
              + " representation. Ensure that the input is a JSON-serialized Pandas DataFrame"
              + " with the `split` orientation.",
          e);
    } catch (InvalidSchemaException e) {
      logger.error(
          "Encountered a schema mismatch while transforming input into a Pandas DataFrame"
              + " representation.",
          e);
      throw new PredictorEvaluationException(
          "Encountered a schema mismatch while transforming input into a Pandas DataFrame"
              + " representation. Ensure that the input is a JSON-serialized Pandas DataFrame"
              + " with the `split` orientation.",
          e);
    } catch (IllegalArgumentException e) {
      logger.error(
          "Failed to transform input into a Pandas DataFrame because the parsed frame is invalid.",
          e);
      throw new PredictorEvaluationException(
          "Failed to transform input into a Pandas DataFrame because the parsed frame is invalid."
              + " Ensure that the input is a JSON-serialized Pandas DataFrame with the `split`"
              + " orientation.",
          e);
    }

    DefaultLeapFrame leapFrame;
    try {
      leapFrame = pandasFrame.toLeapFrame(this.inputSchema);
    } catch (InvalidSchemaException e) {
      logger.error(
          "Encountered a schema mismatch when converting the input dataframe to a LeapFrame.", e);
      throw new PredictorEvaluationException(
          "Encountered a schema mismatch when converting the input dataframe to a LeapFrame.");
    } catch (Exception e) {
      logger.error(
          "Encountered an unknown error during conversion of Pandas dataframe to LeapFrame.", e);
      throw new PredictorEvaluationException(
          "An unknown error occurred while converting the input dataframe to a LeapFrame.", e);
    }
    // Create a single-element sequence of column names to select from the resulting dataframe.
    // This single-element is the `prediction` column; as is the case with the `pyfunc` wrapper
    // for Spark models, the query response is comprised solely of entries in the `prediction`
    // column
    DefaultLeapFrame predictionsFrame = this.leapFrameSupport.select(
        this.pipelineTransformer
            .transform(leapFrame)
            .get(), Collections.singletonList(PREDICTION_COLUMN_NAME));

    List<Object> predictions = this.leapFrameSupport.collect(predictionsFrame)
            .stream()
            .map(row -> row.getRaw(0))
            .collect(Collectors.toList());

    try {
      String predictionsJson = SerializationUtils.toJson(predictions);
      return new PredictorDataWrapper(predictionsJson, PredictorDataWrapper.ContentType.Json);
    } catch (JsonProcessingException e) {
      logger.error("Encountered an error while serializing the output dataframe.", e);
      throw new PredictorEvaluationException(
          "Failed to serialize prediction results as a JSON list!");
    }
  }

  /** @return The underlying MLeap pipeline transformer that this predictor uses for inference */
  public Transformer getPipeline() {
    return this.pipelineTransformer;
  }
}
