package com.databricks.mlflow;

import com.databricks.mlflow.models.Model;
import com.databricks.mlflow.models.Predictor;
import com.databricks.mlflow.Flavor;

import java.util.Optional;

public abstract class LoaderModule<T extends Flavor> {
    public Predictor load(Model modelConfig) {
        Optional<T> flavor = modelConfig.getFlavor(getFlavorName(), getFlavorClass());
        if (!flavor.isPresent()) {
            // throw new Exception();
        }
        return createPredictor(flavor.get().getModelDataPath());
    }

    protected abstract Predictor createPredictor(String modelDataPath);

    protected abstract Class<T> getFlavorClass();

    protected abstract String getFlavorName();
}
