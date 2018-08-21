package org.mlflow.sagemaker;

import java.nio.charset.Charset;
import ml.combust.mleap.json.DefaultFrameReader;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;

/**
 * Utilities for serializing, deserialize, and manipulating MLeap {@link
 * ml.combust.mleap.runtime.frame.LeapFrame} objects
 */
class LeapFrameUtils {
  private static final DefaultFrameReader frameReader = new DefaultFrameReader();
  private static final Charset jsonCharset = Charset.forName("UTF-8");

  /**
   * Deserializes a {@link ml.combust.mleap.runtime.frame.LeapFrame} from its serialized JSON
   * representation
   */
  protected static DefaultLeapFrame getLeapFrameFromJson(String frameJson) {
    byte[] frameBytes = frameJson.getBytes(jsonCharset);
    return frameReader.fromBytes(frameBytes, jsonCharset).get();
  }
}
