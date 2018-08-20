package org.mlflow.mleap;

import java.nio.charset.Charset;

import ml.combust.mleap.json.DefaultFrameReader;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;

public class LeapFrameUtils {
    private static final DefaultFrameReader frameReader = new DefaultFrameReader();
    private static final Charset jsonCharset = Charset.forName("UTF-8");

    public static DefaultLeapFrame getLeapFrameFromJson(String inputJson) {
        byte[] inputBytes = inputJson.getBytes(jsonCharset);
        return frameReader.fromBytes(inputBytes, jsonCharset).get();
    }
}
