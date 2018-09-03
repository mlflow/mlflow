package org.mlflow.tracking.creds;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

public class DatabricksConfigHostCredsProviderTest {
  private String previousUserHome = null;
  private File databrickscfg = null;

  @BeforeSuite
  public void beforeAll() throws IOException {
    previousUserHome = System.getProperty("user.home");
    Path tempDir = Files.createTempDirectory(getClass().getSimpleName());
    databrickscfg = tempDir.resolve(".databrickscfg").toFile();
    System.setProperty("user.home", tempDir.toString());
  }

  @AfterSuite
  public void afterAll() {
    if (previousUserHome != null) {
      System.setProperty("user.home", previousUserHome);
    }
  }

  @Test
  public void testGetTokenFromDefault() throws IOException {
    String contents = "[DEFAULT]\nhost = https://boop.com\ntoken = dapi\n";
    FileUtils.writeStringToFile(databrickscfg, contents, StandardCharsets.UTF_8);
    DatabricksConfigHostCredsProvider provider = new DatabricksConfigHostCredsProvider();
    Assert.assertEquals(provider.getHostCreds().getHost(), "https://boop.com");
    Assert.assertEquals(provider.getHostCreds().getToken(), "dapi");

    String contents2 = "[DEFAULT]\nhost = https://boop.com\ntoken=dapi2\ninsecure = TrUe";
    FileUtils.writeStringToFile(databrickscfg, contents2, StandardCharsets.UTF_8);
    Assert.assertEquals(provider.getHostCreds().getToken(), "dapi");
    Assert.assertFalse(provider.getHostCreds().shouldIgnoreTlsVerification());
    provider.refresh();
    Assert.assertEquals(provider.getHostCreds().getToken(), "dapi2");
    Assert.assertTrue(provider.getHostCreds().shouldIgnoreTlsVerification());
  }

  @Test
  public void testGetUserPassFromProfile() throws IOException {
    String contents = "[myprof]\nhost = https://boop.com\nusername = Bob\npassword = Ross\n";
    FileUtils.writeStringToFile(databrickscfg, contents, StandardCharsets.UTF_8);
    DatabricksConfigHostCredsProvider provider = new DatabricksConfigHostCredsProvider("myprof");
    Assert.assertEquals(provider.getHostCreds().getHost(), "https://boop.com");
    Assert.assertEquals(provider.getHostCreds().getUsername(), "Bob");
    Assert.assertEquals(provider.getHostCreds().getPassword(), "Ross");

    try {
      new DatabricksConfigHostCredsProvider().getHostCreds();
      Assert.fail();
    } catch (IllegalStateException e) {
      Assert.assertTrue(e.getMessage().contains("Could not find 'DEFAULT'"), e.getMessage());
    }

    try {
      new DatabricksConfigHostCredsProvider("blah").getHostCreds();
      Assert.fail();
    } catch (IllegalStateException e) {
      Assert.assertTrue(e.getMessage().contains("Could not find 'blah'"), e.getMessage());
    }
  }

  @Test
  public void testProfileNoHost() throws IOException {
    String contents = "[DEFAULT]\ntoken = dabi\n";
    FileUtils.writeStringToFile(databrickscfg, contents, StandardCharsets.UTF_8);
    try {
      new DatabricksConfigHostCredsProvider().getHostCreds();
      Assert.fail();
    } catch (IllegalStateException e) {
      Assert.assertTrue(e.getMessage().contains("No 'host' configured"), e.getMessage());
    }
  }

  @Test
  public void testProfileNoAuth() throws IOException {
    String contents = "[DEFAULT]\nhost = foo\n";
    FileUtils.writeStringToFile(databrickscfg, contents, StandardCharsets.UTF_8);
    try {
      new DatabricksConfigHostCredsProvider().getHostCreds();
      Assert.fail();
    } catch (IllegalStateException e) {
      Assert.assertTrue(e.getMessage().contains("No authentication configured"), e.getMessage());
    }
  }

  @Test
  public void testProfileNoFile() throws IOException {
    databrickscfg.delete();
    try {
      new DatabricksConfigHostCredsProvider().getHostCreds();
      Assert.fail();
    } catch (IllegalStateException e) {
      Assert.assertTrue(e.getMessage().contains("Could not find Databricks configuration file"),
        e.getMessage());
    }
  }
}
