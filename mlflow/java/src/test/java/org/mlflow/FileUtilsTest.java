package org.mlflow.utils;

import org.junit.Assert;
import org.junit.Test;

public class FileUtilsTest {
  @Test
  public void testFilePathsAreConcatenatedCorrectly() {
    String outputPathRelative = "this/is/an/mlflow/test/path";

    String joinedPath1 = FileUtils.join("this", "is/", "an", "mlflow/", "test", "path/");
    String joinedPath2 = FileUtils.join("this/", "is", "an", "mlflow", "test/", "path");

    Assert.assertEquals(joinedPath1, outputPathRelative);
    Assert.assertEquals(joinedPath2, outputPathRelative);

    String outputPathAbsolute = "/this/is/an/mlflow/test/path";

    String joinedPath3 = FileUtils.join("/this", "is/", "an", "mlflow/", "test", "path/");
    String joinedPath4 = FileUtils.join("/this/", "is", "an", "mlflow", "test/", "path");

    Assert.assertEquals(joinedPath3, outputPathAbsolute);
    Assert.assertEquals(joinedPath4, outputPathAbsolute);
  }
}
