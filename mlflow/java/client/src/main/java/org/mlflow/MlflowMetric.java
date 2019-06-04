package org.mlflow;

public class MlflowMetric {
    public String key;
    public double value;
    public MlflowMetric(String key, double value) {
        this.key = key;
        this.value = value;
    }
}
