package org.mlflow.sagemaker;

import java.nio.charset.Charset;

import ml.combust.mleap.json.DefaultFrameReader;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;

class LeapFrameUtils {
    private static final DefaultFrameReader frameReader = new DefaultFrameReader();
    private static final Charset jsonCharset = Charset.forName("UTF-8");

    protected static DefaultLeapFrame getLeapFrameFromJson(String frameJson) {
        byte[] frameBytes = frameJson.getBytes(jsonCharset);
        return frameReader.fromBytes(frameBytes, jsonCharset).get();
    }
}
