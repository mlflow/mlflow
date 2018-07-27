package com.databricks.mlflow.mleap;

import com.databricks.mlflow.models.ScalaModel;
import com.databricks.mlflow.models.ModelConfig;

import java.util.Optional;
import java.io.IOException;

public class MLeapModel extends ScalaModel {
    private final MLeapPredictor predictor;

    public MLeapModel(String modelDataPath) {
        this.predictor = new MLeapPredictor(modelDataPath);
    }

    public static MLeapModel fromConfig(String configPath) {
        ModelConfig modelConfig = null;
        try {
            modelConfig = ModelConfig.fromPath(configPath);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        Optional<MLeapConfig> mleapConfig =
            modelConfig.getFlavor(MLeapConfig.FLAVOR_NAME, MLeapConfig.class);
        if (!mleapConfig.isPresent()) {
            // Raise exception
        }
        return new MLeapModel(mleapConfig.get().getModelDataPath());
    }

    public static MLeapModel load(String path) {
        return new MLeapModel(path);
    }

    @Override
    public String predict(String input) {
        return this.predictor.predict(input);
    }

    public static void main(String[] args) {
        String path = args[0];
        MLeapModel.fromConfig(path);
    }
}
