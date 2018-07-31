package com.databricks.mlflow;

import com.databricks.mlflow.models.Model;
import com.databricks.mlflow.models.Predictor;
import com.databricks.mlflow.Flavor;

import java.util.Optional;
import java.io.File;

public abstract class LoaderModule<T extends Flavor> {
    public Predictor load(Model modelConfig) {
        Optional<T> flavor = modelConfig.getFlavor(getFlavorName(), getFlavorClass());
        if (!flavor.isPresent()) {
            // throw new Exception();
        }
        Optional<String> basePath = modelConfig.getBasePath();
        if (!basePath.isPresent()) {
            // TODO: Raise exception!
            // throw new Exception();
        }
        String absoluteModelPath =
            basePath.get() + File.separator + flavor.get().getModelDataPath();
        return createPredictor(absoluteModelPath);
    }

    protected abstract Predictor createPredictor(String modelDataPath);

    protected abstract Class<T> getFlavorClass();

    protected abstract String getFlavorName();
}
