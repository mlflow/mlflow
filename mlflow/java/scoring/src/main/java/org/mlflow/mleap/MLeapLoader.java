package org.mlflow.mleap;

import java.io.IOException;
import ml.combust.mleap.runtime.frame.Transformer;
import org.mlflow.LoaderModule;
import org.mlflow.models.Model;
import org.mlflow.sagemaker.MLeapPredictor;
import org.mlflow.sagemaker.PredictorLoadingException;
import org.mlflow.utils.FileUtils;

public class MLeapLoader extends LoaderModule<MLeapFlavor> {
  /** Loads an MLflow model with the MLeap flavor as an MLeap transformer */
  public Transformer loadPipeline(String modelRootPath) throws PredictorLoadingException {
    try {
      return ((MLeapPredictor) load(Model.fromRootPath(modelRootPath))).getPipeline();
    } catch (IOException e) {
      throw new PredictorLoadingException(
          String.format(
              "Failed to read model files from the supplied model root path: %s."
                  + "Please ensure that this is the path to a valid MLflow model.",
              modelRootPath));
    }
  }

  /**
   * Loads an MLflow model with the MLeap flavor as a generic predictor that can be used for
   * inference
   */
  @Override
  protected MLeapPredictor createPredictor(String modelRootPath, MLeapFlavor flavor) {
    String modelDataPath = FileUtils.join(modelRootPath, flavor.getModelDataPath());
    return new MLeapPredictor(modelDataPath);
  }

  @Override
  protected Class<MLeapFlavor> getFlavorClass() {
    return MLeapFlavor.class;
  }

  @Override
  protected String getFlavorName() {
    return MLeapFlavor.FLAVOR_NAME;
  }
}
