package org.mlflow.artifacts;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Sets;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

import org.mlflow.api.proto.Service.FileInfo;
import org.mlflow.api.proto.Service.RunInfo;
import org.mlflow.tracking.MlflowClient;
import org.mlflow.tracking.TestClientProvider;
import org.mlflow.tracking.creds.BasicMlflowHostCreds;
import org.mlflow.tracking.creds.DatabricksMlflowHostCreds;
import org.mlflow.tracking.creds.MlflowHostCreds;

public class CliBasedArtifactRepositoryTest {
  private static final Logger logger = LoggerFactory.getLogger(
    CliBasedArtifactRepositoryTest.class);

  private final TestClientProvider testClientProvider = new TestClientProvider();

  private MlflowClient client;

  @BeforeSuite
  public void beforeAll() throws IOException {
    client = testClientProvider.initializeClientAndServer();
  }

  @AfterSuite
  public void afterAll() throws InterruptedException {
    testClientProvider.cleanupClientAndServer();
  }

  private CliBasedArtifactRepository newRepo() {
    RunInfo runInfo = client.createRun();
    logger.info("Created run with id=" + runInfo.getRunUuid() + " and artifactUri=" +
      runInfo.getArtifactUri());
    return new CliBasedArtifactRepository(runInfo.getArtifactUri(), runInfo.getRunUuid(),
      testClientProvider.getClientHostCredsProvider(client));
  }

  @Test
  public void testLogAndDownloadArtifact() throws IOException {
    // Tests the logArtifact and downloadArtifacts APIs when targeting a single file.
    ArtifactRepository repo = newRepo();
    Path tempFile = Files.createTempFile(getClass().getSimpleName(), ".txt");
    FileUtils.writeStringToFile(tempFile.toFile(), "Hello, World!", StandardCharsets.UTF_8);
    repo.logArtifact(tempFile.toFile());
    Path returnFile = repo.downloadArtifacts(tempFile.getFileName().toString()).toPath();
    Assert.assertEquals(readFile(returnFile), "Hello, World!");
  }

  @Test
  public void testLogListAndDownloadArtifacts() throws IOException {
    // Tests the logArtifacts, list, downloadArtifacts APIs when targeting a set of files.
    ArtifactRepository repo = newRepo();
    Path tempDir = Files.createTempDirectory(getClass().getSimpleName());
    FileUtils.writeStringToFile(tempDir.resolve("a").toFile(), "A", StandardCharsets.UTF_8);
    FileUtils.writeStringToFile(tempDir.resolve("b").toFile(), "B", StandardCharsets.UTF_8);
    repo.logArtifacts(tempDir.toFile());
    Set<FileInfo> fileInfos = Sets.newHashSet(repo.listArtifacts());
    Assert.assertEquals(fileInfos, Sets.newHashSet(fileInfo("a", 1), fileInfo("b", 1)));
    Path returnDir = repo.downloadArtifacts().toPath();
    Assert.assertEquals(readFile(returnDir.resolve("a")), "A");
    Assert.assertEquals(readFile(returnDir.resolve("b")), "B");
  }

  @Test
  public void testEverything() throws IOException {
    // This is a comprehensive integration test which tests all 8 APIs exposed by ArtifactRepository
    // on a mix of files, directories, and subdirectories.
    ArtifactRepository repo = newRepo();

    // Create a temporary directory with /childFile and /childDir/granchild as files.
    Path tempDir = Files.createTempDirectory(getClass().getSimpleName());
    Path childFile = tempDir.resolve("childFile");
    String childContents = "File contents!";
    FileUtils.writeStringToFile(childFile.toFile(), childContents, StandardCharsets.UTF_8);
    Path childDir = tempDir.resolve("childDir");
    Path grandchildFile = childDir.resolve("grandchild");
    String grandchildContents = "Baby content!";
    childDir.toFile().mkdir();
    FileUtils.writeStringToFile(grandchildFile.toFile(), grandchildContents, StandardCharsets.UTF_8);

    // Log artifacts such that we expect:
    //   childFile
    //   grandchild
    //   subpath/childFile
    //   subpath/subberpath/grandchild
    repo.logArtifact(childFile.toFile());
    repo.logArtifacts(childDir.toFile());
    repo.logArtifact(childFile.toFile(), "subpath");
    repo.logArtifacts(childDir.toFile(), "subpath/subberpath");

    // List at the root, we should see childFile, grandchild, and subpath/.
    List<FileInfo> fileInfos = repo.listArtifacts();
    Set<FileInfo> expectedFileInfos = Sets.newHashSet(
      fileInfo("childFile", childContents.length()),
      fileInfo("grandchild", grandchildContents.length()),
      dirInfo("subpath")
    );
    Assert.assertEquals(Sets.newHashSet(fileInfos), expectedFileInfos);

    // List within subpath/, we should see childFile and subberpath.
    List<FileInfo> subpathFileInfos = repo.listArtifacts("subpath");
    Set<FileInfo> expectedSubpathFileInfos = Sets.newHashSet(
      fileInfo("subpath/childFile", childContents.length()),
      dirInfo("subpath/subberpath")
    );
    Assert.assertEquals(Sets.newHashSet(subpathFileInfos), expectedSubpathFileInfos);

    // Download everything, and confirm that we have the four expected files.
    Path allArtifacts = repo.downloadArtifacts().toPath();
    Assert.assertEquals(childContents, readFile(allArtifacts.resolve("childFile")));
    Assert.assertEquals(childContents, readFile(allArtifacts.resolve("subpath/childFile")));
    Assert.assertEquals(grandchildContents, readFile(allArtifacts.resolve("grandchild")));
    Assert.assertEquals(grandchildContents,
      readFile(allArtifacts.resolve("subpath/subberpath/grandchild")));

    // Download subpath/subberpath, and confirm that we have just the grandchild.
    Path subberpathArtifacts = repo.downloadArtifacts("subpath/subberpath").toPath();
    Assert.assertEquals(grandchildContents, readFile(subberpathArtifacts.resolve("grandchild")));
    Assert.assertEquals(subberpathArtifacts.toFile().list(), new String[] {"grandchild"});
  }

  @Test
  public void testSettingProcessEnvBasic() {
    CliBasedArtifactRepository repo = newRepo();
    MlflowHostCreds hostCreds = new BasicMlflowHostCreds("just-host");
    Map<String, String> env = new HashMap<>();
    repo.setProcessEnvironment(env, hostCreds);
    Map<String, String> expectedEnv = new HashMap<>();
    expectedEnv.put("MLFLOW_TRACKING_URI", "just-host");
    Assert.assertEquals(env, expectedEnv);
  }

  @Test
  public void testSettingProcessEnvUserPass() {
    CliBasedArtifactRepository repo = newRepo();
    MlflowHostCreds hostCreds = new BasicMlflowHostCreds("just-host", "user", "pass");
    Map<String, String> env = new HashMap<>();
    repo.setProcessEnvironment(env, hostCreds);
    Map<String, String> expectedEnv = new HashMap<>();
    expectedEnv.put("MLFLOW_TRACKING_URI", "just-host");
    expectedEnv.put("MLFLOW_TRACKING_USERNAME", "user");
    expectedEnv.put("MLFLOW_TRACKING_PASSWORD", "pass");
    Assert.assertEquals(env, expectedEnv);
  }

  @Test
  public void testSettingProcessEnvToken() {
    CliBasedArtifactRepository repo = newRepo();
    MlflowHostCreds hostCreds = new BasicMlflowHostCreds("just-host", "token");
    Map<String, String> env = new HashMap<>();
    repo.setProcessEnvironment(env, hostCreds);
    Map<String, String> expectedEnv = new HashMap<>();
    expectedEnv.put("MLFLOW_TRACKING_URI", "just-host");
    expectedEnv.put("MLFLOW_TRACKING_TOKEN", "token");
    Assert.assertEquals(env, expectedEnv);
  }

  @Test
  public void testSettingProcessEnvInsecure() {
    CliBasedArtifactRepository repo = newRepo();
    MlflowHostCreds hostCreds = new BasicMlflowHostCreds("insecure-host", null, null, null,
      true);
    Map<String, String> env = new HashMap<>();
    repo.setProcessEnvironment(env, hostCreds);
    Map<String, String> expectedEnv = new HashMap<>();
    expectedEnv.put("MLFLOW_TRACKING_URI", "insecure-host");
    expectedEnv.put("MLFLOW_TRACKING_INSECURE_TLS", "true");
    Assert.assertEquals(env, expectedEnv);
  }

  @Test
  public void testSettingProcessEnvDatabricksUserPass() {
    CliBasedArtifactRepository repo = newRepo();
    DatabricksMlflowHostCreds hostCreds = new DatabricksMlflowHostCreds(
      "just-host", "user", "pass");
    Map<String, String> env = new HashMap<>();
    repo.setProcessEnvironmentDatabricks(env, hostCreds);
    Map<String, String> expectedEnv = new HashMap<>();
    expectedEnv.put("DATABRICKS_HOST", "just-host");
    expectedEnv.put("DATABRICKS_USERNAME", "user");
    expectedEnv.put("DATABRICKS_PASSWORD", "pass");
    Assert.assertEquals(env, expectedEnv);
  }

  @Test
  public void testSettingProcessEnvDatabricksToken() {
    CliBasedArtifactRepository repo = newRepo();
    DatabricksMlflowHostCreds hostCreds = new DatabricksMlflowHostCreds("just-host", "token");
    Map<String, String> env = new HashMap<>();
    repo.setProcessEnvironmentDatabricks(env, hostCreds);
    Map<String, String> expectedEnv = new HashMap<>();
    expectedEnv.put("DATABRICKS_HOST", "just-host");
    expectedEnv.put("DATABRICKS_TOKEN", "token");
    Assert.assertEquals(env, expectedEnv);
  }

  @Test
  public void testSettingProcessEnvDatabricksInsecure() {
    CliBasedArtifactRepository repo = newRepo();
    DatabricksMlflowHostCreds hostCreds = new DatabricksMlflowHostCreds(
      "insecure-host", null, null, null, true);
    Map<String, String> env = new HashMap<>();
    repo.setProcessEnvironmentDatabricks(env, hostCreds);
    Map<String, String> expectedEnv = new HashMap<>();
    expectedEnv.put("DATABRICKS_HOST", "insecure-host");
    expectedEnv.put("DATABRICKS_INSECURE", "true");
    Assert.assertEquals(env, expectedEnv);
  }

  private String readFile(Path path) throws IOException {
    return FileUtils.readFileToString(path.toFile(), StandardCharsets.UTF_8);
  }

  private FileInfo fileInfo(String path, int fileSize) {
    return FileInfo.newBuilder().setPath(path).setFileSize(fileSize).setIsDir(false).build();
  }
  private FileInfo dirInfo(String path) {
    return FileInfo.newBuilder().setPath(path).setIsDir(true).build();
  }
}
