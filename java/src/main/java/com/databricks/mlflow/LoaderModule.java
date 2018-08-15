package com.databricks.mlflow;

import com.databricks.mlflow.models.Model;
import com.databricks.mlflow.sagemaker.Predictor;
import com.databricks.mlflow.Flavor;

import java.util.Optional;

public abstract class LoaderModule<T extends Flavor> {
    /**
     * Loads an MLFlow model as a generic predictor that can be used for
     * inference
     *
     * @throws PredictorLoadingException Thrown if failures are encountered while attempting to load
     * the model from the specified configuration. This is a generic exception for all loader
     * failure cases
     */
    public Predictor load(Model modelConfig) {
        Optional<T> flavor = modelConfig.getFlavor(getFlavorName(), getFlavorClass());
        if (!flavor.isPresent()) {
            throw new PredictorLoadingException(
                String.format("Attempted to load the %s flavor of the model,"
                        + " but the model does not have this flavor.",
                    getFlavorName()));
        }
        Optional<String> rootPath = modelConfig.getRootPath();
        if (!rootPath.isPresent()) {
            throw new PredictorLoadingException(
                "An internal error occurred while loading the model:"
                + " the model's root path could not be found. Please ensure that this"
                + " model was created using `Model.fromRootPath()` or `Model.fromConfigPath()`");
        }
        return createPredictor(rootPath.get(), flavor.get());
    }

    /**
     * Implementations of this method are expected to throw a `PredictorLoadingException`
     * when errors are encountered while loading the model
     */
    protected abstract Predictor createPredictor(String modelRootPath, T flavor)
        throws PredictorLoadingException;

    protected abstract Class<T> getFlavorClass();

    protected abstract String getFlavorName();
}
