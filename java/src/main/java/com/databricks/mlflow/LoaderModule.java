package com.databricks.mlflow;

import com.databricks.mlflow.models.Model;
import com.databricks.mlflow.sagemaker.Predictor;
import com.databricks.mlflow.Flavor;

import java.util.Optional;

public abstract class LoaderModule<T extends Flavor> {
    public Predictor load(Model modelConfig) throws LoaderModuleException {
        Optional<T> flavor = modelConfig.getFlavor(getFlavorName(), getFlavorClass());
        if (!flavor.isPresent()) {
            throw new LoaderModuleException(
                String.format("Attempted to load the %s flavor of the model,"
                        + " but the model does not have this flavor.",
                    getFlavorName()));
        }
        Optional<String> rootPath = modelConfig.getRootPath();
        if (!rootPath.isPresent()) {
            throw new LoaderModuleException("An internal error occurred while loading the model:"
                + " the model's root path could not be found. Please ensure that this"
                + " model was created using `Model.fromRootPath()` or `Model.fromConfigPath()`");
        }
        return createPredictor(rootPath.get(), flavor.get());
    }

    protected abstract Predictor createPredictor(String modelRootPath, T flavor)
        throws LoaderModuleException;

    protected abstract Class<T> getFlavorClass();

    protected abstract String getFlavorName();
}
