package org.mlflow.sagemaker;

import com.fasterxml.jackson.core.JsonProcessingException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import ml.combust.mleap.runtime.frame.Transformer;
import org.junit.Assert;
import org.junit.Test;
import org.mlflow.MLflowRootResourceProvider;
import org.mlflow.mleap.MLeapLoader;
import org.mlflow.utils.SerializationUtils;

/** Unit tests for the {@link MLeapPredictor} */
public class MLeapPredictorTest {
  @Test
  public void testMLeapPredictorGetPipelineYieldsValidMLeapTransformer()
      throws PredictorLoadingException {
    String modelPath = MLflowRootResourceProvider.getResourcePath("mleap_model");
    MLeapPredictor predictor = (MLeapPredictor) (new MLeapLoader()).load(modelPath);
    Transformer pipelineTransformer = predictor.getPipeline();
  }

  @Test
  public void testMLeapPredictorEvaluatesCompatibleInputCorrectly()
      throws IOException, PredictorEvaluationException {
    String modelPath = MLflowRootResourceProvider.getResourcePath("mleap_model");
    MLeapPredictor predictor = (MLeapPredictor) (new MLeapLoader()).load(modelPath);

    String sampleInputPath =
        MLflowRootResourceProvider.getResourcePath("mleap_model/sample_input.json");
    String sampleInputJson = new String(Files.readAllBytes(Paths.get(sampleInputPath)));
    PredictorDataWrapper inputData =
        new PredictorDataWrapper(sampleInputJson, PredictorDataWrapper.ContentType.Json);
    PredictorDataWrapper outputData = predictor.predict(inputData);
  }

  @Test
  public void
  testMLeapPredictorThrowsPredictorEvaluationExceptionWhenEvaluatingInputWithMissingField()
      throws IOException, JsonProcessingException {
    String modelPath = MLflowRootResourceProvider.getResourcePath("mleap_model");
    MLeapPredictor predictor = (MLeapPredictor) (new MLeapLoader()).load(modelPath);

    String sampleInputPath =
        MLflowRootResourceProvider.getResourcePath("mleap_model/sample_input.json");
    String sampleInputJson = new String(Files.readAllBytes(Paths.get(sampleInputPath)));
    List<Map<String, Object>> sampleInput =
        SerializationUtils.fromJson(sampleInputJson, List.class);

    sampleInput.get(0).remove("topic");
    String badInputJson = SerializationUtils.toJson(sampleInput);
    PredictorDataWrapper inputData =
        new PredictorDataWrapper(badInputJson, PredictorDataWrapper.ContentType.Json);
    try {
      predictor.predict(inputData);
      Assert.fail("Expected predictor evaluation on a dataframe with a missing field to fail.");
    } catch (PredictorEvaluationException e) {
      // Success
    }
  }

  @Test
  /**
   * NOTE: When PredictorDataWrapper objects start performing JSON format validation, this test will
   * need to be updated to ensure that bad JSON is still being passed to the {@link MLeapPredictor}
   */
  public void testMLeapPredictorThrowsPredictorEvaluationExceptionWhenEvaluatingBadJson() {
    String modelPath = MLflowRootResourceProvider.getResourcePath("mleap_model");
    MLeapPredictor predictor = (MLeapPredictor) (new MLeapLoader()).load(modelPath);

    String badInputJson = "This is not a valid json string";
    PredictorDataWrapper badInputData =
        new PredictorDataWrapper(badInputJson, PredictorDataWrapper.ContentType.Json);

    try {
      predictor.predict(badInputData);
      Assert.fail("Expected predictor evaluation on a bad JSON input"
          + "to throw a PredictorEvaluationException.");
    } catch (PredictorEvaluationException e) {
      // Success
    }
  }
}
