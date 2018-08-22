package org.mlflow.utils;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.junit.Assert;
import org.junit.Test;

public class FileUtilsTest {
  @Test
  public void testFilePathsAreConcatenatedCorrectly() {
    String outputPathRelative = "this/is/an/mlflow/test/path";

    String joinedPath1 = FileUtils.join("this", "is/", "an", "mlflow/", "test", "path/");
    String joinedPath2 = FileUtils.join("this/", "is", "an", "mlflow", "test/", "path");

    Assert.assertEquals(outputPathRelative, joinedPath1);
    Assert.assertEquals(outputPathRelative, joinedPath2);

    String outputPathAbsolute = "/this/is/an/mlflow/test/path";

    String joinedPath3 = FileUtils.join("/this", "is/", "an", "mlflow/", "test", "path/");
    String joinedPath4 = FileUtils.join("/this/", "is", "an", "mlflow", "test/", "path");

    Assert.assertEquals(outputPathAbsolute, joinedPath3);
    Assert.assertEquals(outputPathAbsolute, joinedPath4);
  }

  @Test
  public void utf8ContentIsReadFromInputStreamCorrectly() throws IOException {
    String sampleString = "This is a sample string.";
    InputStream stream = new ByteArrayInputStream(sampleString.getBytes(StandardCharsets.UTF_8));
    String parsedString = FileUtils.readInputStreamAsUtf8(stream);
    Assert.assertEquals(sampleString, parsedString);
  }
}
