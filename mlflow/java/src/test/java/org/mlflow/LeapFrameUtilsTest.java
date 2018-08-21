package org.mlflow.sagemaker;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import ml.combust.mleap.runtime.frame.DefaultLeapFrame;
import org.junit.Test;

/** Unit tests for the {@link LeapFrameUtils} module */
public class LeapFrameUtilsTest {
  @Test
  public void testValidSerializedLeapFrameIsDeserializedAsLeapFrameObjectSuccessfully()
      throws IOException {
    String framePath = getClass().getResource("sample_leapframe.json").getFile();
    String frameJson = new String(Files.readAllBytes(Paths.get(framePath)));
    DefaultLeapFrame leapFrame = LeapFrameUtils.getLeapFrameFromJson(frameJson);
  }
}
