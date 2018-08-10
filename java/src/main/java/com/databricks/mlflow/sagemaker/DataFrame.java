package com.databricks.mlflow.sagemaker;

import com.databricks.mlflow.mleap.LeapFrameUtils;

import ml.combust.mleap.runtime.frame.DefaultLeapFrame;

public class DataFrame {
    private final DefaultLeapFrame leapFrame;

    protected DataFrame(DefaultLeapFrame leapFrame) {
        this.leapFrame = leapFrame;
    }

    public DefaultLeapFrame getLeapFrame() {
        return this.leapFrame;
    }

    protected String toJson() {
        return LeapFrameUtils.getJsonFromLeapFrame(this.leapFrame);
    }

    protected String toCsv() {
        return LeapFrameUtils.getCsvFromLeapFrame(this.leapFrame);
    }

    protected static DataFrame fromJson(String frameJson) {
        return new DataFrame(LeapFrameUtils.getLeapFrameFromJson(frameJson));
    }

    protected static DataFrame fromCsv(String frameJson) {
        return new DataFrame(LeapFrameUtils.getLeapFrameFromCsv(frameJson));
    }
}
