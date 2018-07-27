package com.databricks.mlflow;

import java.util.Optional;

public class MLeapModel extends ScalaModel {
    private final ScalaModel predictor;

    public MLeapModel(String modelDataPath) {
        this.predictor = new MLeapPredictor(modelDataPath);
    }

    public static MLeapModel fromConfig(String configPath) {
        ModelConfig modelConfig = ModelConfig.fromPath(configPath);
        Optional<MLeapConfig> mleapConfig =
            modelConfig.getFlavor(MLeapConfig.FLAVOR_NAME, MLeapConfig.class);
        if (!mleapConfig.isPresent()) {
            // Raise exception
        }
        // Check that versions match
        return new MLeapModel(mleapConfig.get().getModelDataPath());
    }

    @Override
    public String predict(String input) {
        return null;
    }

    public static void main(String[] args) {
        String path = args[0];
        MLeapModel.fromConfig(path);
    }
}
