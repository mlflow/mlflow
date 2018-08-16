package com.databricks.mlflow.mleap;

import com.databricks.mlflow.LoaderModule;
import com.databricks.mlflow.models.Model;
import com.databricks.mlflow.sagemaker.MLeapPredictor;
import com.databricks.mlflow.sagemaker.Predictor;
import com.databricks.mlflow.sagemaker.PredictorLoadingException;
import com.databricks.mlflow.utils.FileUtils;

import java.util.Optional;
import java.io.IOException;

import ml.combust.mleap.runtime.frame.Transformer;

public class MLeapLoader extends LoaderModule<MLeapFlavor> {
    /**
     * Loads an MLFlow model with the MLeap flavor as an MLeap transformer
     */
    public Transformer loadPipeline(String modelRootPath) throws PredictorLoadingException {
        try {
            return ((MLeapPredictor) load(Model.fromRootPath(modelRootPath))).getPipeline();
        } catch (IOException e) {
            throw new PredictorLoadingException(
                String.format("Failed to read model files from the supplied model root path: %s."
                        + "Please ensure that this is the path to a valid MLFlow model.",
                    modelRootPath));
        }
    }

    /**
     * Loads an MLFlow model with the MLeap flavor as a generic predictor that can be used
     * for inference
     */
    @Override
    protected Predictor createPredictor(String modelRootPath, MLeapFlavor flavor) {
        String modelDataPath = FileUtils.join(modelRootPath, flavor.getModelDataPath());
        String inputSchemaPath = FileUtils.join(modelRootPath, flavor.getInputSchemaPath());
        return new MLeapPredictor(modelDataPath, inputSchemaPath);
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
