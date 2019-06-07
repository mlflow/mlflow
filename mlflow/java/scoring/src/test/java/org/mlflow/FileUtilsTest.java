package org.mlflow.utils;

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
}
