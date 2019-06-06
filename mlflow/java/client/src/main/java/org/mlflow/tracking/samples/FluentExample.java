package org.mlflow.tracking.samples;

import org.mlflow.MlflowMetric;
import org.mlflow.tracking.ActiveRun;
import org.mlflow.tracking.MlflowTrackingContext;

import java.util.Arrays;
import java.util.Collections;

public class FluentExample {
    public static void main(String[] args) {
        MlflowTrackingContext mlflow = new MlflowTrackingContext();
        ActiveRun run = mlflow.startRun("run");
        run.logParam("alpha", "0.0");
        run.logMetric("MSE", 0.0);
        run.logMetrics(Arrays.asList(
                new MlflowMetric("MSE", 1.0),
                new MlflowMetric("MAE", 1.0)
        ));
        mlflow.endRun();

        mlflow.withActiveRun("apple", (activeRun -> {
            activeRun.logParam("a", "param");
        }));
    }
}
