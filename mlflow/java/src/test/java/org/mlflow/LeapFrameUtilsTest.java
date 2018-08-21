package org.mlflow.sagemaker;

import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Paths;
import java.nio.file.Files;

import ml.combust.mleap.runtime.frame.DefaultLeapFrame;

public class LeapFrameUtilsTest {
  @Test
  public void testValidSerializedLeapFrameIsDeserializedAsLeapFrameObjectSuccessfully()
      throws IOException {
    String framePath = getClass().getResource("sample_leapframe.json").getFile();
    String frameJson = new String(Files.readAllBytes(Paths.get(framePath)));
    DefaultLeapFrame leapFrame = LeapFrameUtils.getLeapFrameFromJson(frameJson);
  }
}
