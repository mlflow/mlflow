package com.databricks.mlflow.mleap;

import com.databricks.mlflow.sagemaker.MLeapPredictor;
import com.databricks.mlflow.sagemaker.Predictor;
import com.databricks.mlflow.LoaderModule;

import java.util.Optional;
import java.io.IOException;

import ml.combust.mleap.runtime.frame.Transformer;

public class MLeapLoader extends LoaderModule<MLeapFlavor> {
    public Transformer loadPipeline(String modelDataPath) {
        return ((MLeapPredictor) createPredictor(modelDataPath)).getPipeline();
    }

    @Override
    protected Predictor createPredictor(String modelDataPath) {
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
