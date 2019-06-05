package org.mlflow.tracking.samples;

import org.mlflow.tracking.ActiveRun;
import org.mlflow.tracking.MlflowTrackingContext;

public class FluentExample {
    public static void main(String[] args) {
        MlflowTrackingContext mlflow = new MlflowTrackingContext();
        ActiveRun run = mlflow.startRun("run");
        run.logParam("alpha", "0.0");
        run.logMetric("MSE", 0.0);
        mlflow.endRun();
    }
}
