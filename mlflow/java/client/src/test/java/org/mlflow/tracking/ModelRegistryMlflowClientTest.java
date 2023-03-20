package org.mlflow.tracking;

import static org.mlflow.tracking.TestUtils.createExperimentName;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;

import com.google.common.collect.Lists;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import org.apache.commons.io.FileUtils;
import org.mlflow.api.proto.ModelRegistry.ModelVersion;
import org.mlflow.api.proto.ModelRegistry.RegisteredModel;
import org.mlflow.api.proto.Service;
import org.mlflow.api.proto.Service.RunInfo;
import org.mockito.Mockito;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class ModelRegistryMlflowClientTest {
    private static final Logger logger = LoggerFactory.getLogger(ModelRegistryMlflowClientTest.class);

    private static final MlflowProtobufMapper mapper = new MlflowProtobufMapper();

    private final TestClientProvider testClientProvider = new TestClientProvider();

    private MlflowClient client;
    private String source;

    private String modelName;

    private static final String content = "Hello, Worldz!";

    // As only a single `.txt` is stored as a model version artifact, this filter is used to
    // extract the written file.
    FilenameFilter filter = new FilenameFilter() {
        @Override
        public boolean accept(File f, String name) {
            return name.endsWith(".txt");
        }
    };

    @BeforeTest
    public void before() throws IOException {
        client = testClientProvider.initializeClientAndSqlLiteBasedServer();
        modelName = "Model-" + UUID.randomUUID().toString();

        String expName = createExperimentName();
        String expId = client.createExperiment(expName);

        RunInfo runCreated = client.createRun(expId);
        String runId = runCreated.getRunUuid();
        source = String.format("runs:/%s/model", runId);

        File tempDir = Files.createTempDirectory("tempDir").toFile();
        File tempFile = Files.createTempFile(tempDir.toPath(), "file", ".txt").toFile();
        FileUtils.writeStringToFile(tempFile, content, StandardCharsets.UTF_8);
        client.logArtifact(runId, tempFile, "model");

        client.sendPost("registered-models/create",
                mapper.makeCreateModel(modelName));

        client.sendPost("model-versions/create",
                mapper.makeCreateModelVersion(modelName, runId, String.format("runs:/%s/model", runId)));
    }

    @AfterTest
    public void after() throws InterruptedException {
        testClientProvider.cleanupClientAndServer();
    }

    @Test
    public void testGetLatestModelVersions() throws IOException {
        // a list of stages
        List<ModelVersion> versions = client.getLatestVersions(modelName,
                Lists.newArrayList("None"));
        Assert.assertEquals(versions.size(), 1);

        validateDetailedModelVersion(versions.get(0), modelName, "None", "1");

        client.sendPatch("model-versions/update", mapper.makeUpdateModelVersion(modelName,
                "1"));
        // get the latest version of all stages
        List<ModelVersion> modelVersion = client.getLatestVersions(modelName);
        Assert.assertEquals(modelVersion.size(), 1);
        validateDetailedModelVersion(modelVersion.get(0), modelName, "None", "1");
        client.sendPost("model-versions/transition-stage",
                mapper.makeTransitionModelVersionStage(modelName, "1", "Staging"));
        modelVersion = client.getLatestVersions(modelName);
        Assert.assertEquals(modelVersion.size(), 1);
        validateDetailedModelVersion(modelVersion.get(0),
                modelName, "Staging", "1");
    }

    @Test
    public void testGetModelVersion() {
        ModelVersion modelVersion = client.getModelVersion(modelName, "1");
        validateDetailedModelVersion(modelVersion, modelName, "Staging", "1");
    }

    @Test(expectedExceptions = MlflowHttpException.class, expectedExceptionsMessageRegExp = ".*RESOURCE_DOES_NOT_EXIST.*")
    public void testGetModelVersion_NotFound() {
        client.getModelVersion(modelName, "2");
    }

    @Test
    public void testGetRegisteredModel() {
	RegisteredModel model = client.getRegisteredModel(modelName);
	Assert.assertEquals(model.getName(), modelName);
	validateDetailedModelVersion(model.getLatestVersions(0), modelName, "Staging", "1" );
    }

    @Test
    public void testGetModelVersionDownloadUri() {
        String downloadUri = client.getModelVersionDownloadUri(modelName, "1");
        Assert.assertEquals(source, downloadUri);
    }

    @Test
    public void testDownloadModelVersion() throws IOException {
        File tempDownloadDir = client.downloadModelVersion(modelName, "1");
        File[] tempDownloadFile = tempDownloadDir.listFiles(filter);
        Assert.assertEquals(tempDownloadFile.length, 1);
        String downloadedContent = FileUtils.readFileToString(tempDownloadFile[0],
                StandardCharsets.UTF_8);
        Assert.assertEquals(content, downloadedContent);
    }

    @Test
    public void testDownloadLatestModelVersion() throws IOException {
        File tempDownloadDir = client.downloadLatestModelVersion(modelName, "None");
        File[] tempDownloadFile = tempDownloadDir.listFiles(filter);
        Assert.assertEquals(tempDownloadFile.length, 1);
        String downloadedContent = FileUtils.readFileToString(tempDownloadFile[0],
                StandardCharsets.UTF_8);
        Assert.assertEquals(content, downloadedContent);
    }

    @Test(expectedExceptions = MlflowClientException.class)
    public void testDownloadLatestModelVersionWhenMoreThanOneVersionIsReturned() {
        MlflowClient mockedClient = Mockito.spy(client);

        List<ModelVersion> modelVersions = Lists.newArrayList();
        modelVersions.add(ModelVersion.newBuilder().build());
        modelVersions.add(ModelVersion.newBuilder().build());
        doReturn(modelVersions).when(mockedClient).getLatestVersions(any(), any());

        mockedClient.downloadLatestModelVersion(modelName, "None");
    }

    private void validateDetailedModelVersion(ModelVersion details, String modelName,
                                              String stage, String version) {
        Assert.assertEquals(details.getCurrentStage(), stage);
        Assert.assertEquals(details.getName(), modelName);
        Assert.assertEquals(details.getVersion(), version);
    }

    @Test
    public void testSearchModelVersions() {
        List<ModelVersion> mvsBefore = client.searchModelVersions().getItems();

        // create new model version of existing registered model
        String newVersionRunId = "newVersionRunId";
        String newVersionSource = "runs:/newVersionRunId/model";
        client.sendPost("model-versions/create",
                mapper.makeCreateModelVersion(modelName, newVersionRunId, newVersionSource));

        // create new registered model
        String modelName2 = "modelName2";
        String runId2 = "runId2";
        String source2 = "runs:/runId2/model";
        client.sendPost("registered-models/create",
                mapper.makeCreateModel(modelName2));
        client.sendPost("model-versions/create",
                mapper.makeCreateModelVersion(modelName2, runId2, source2));

        List<ModelVersion> mvsAfter = client.searchModelVersions().getItems();
        Assert.assertEquals(mvsAfter.size(), 2 + mvsBefore.size());

        String filter1 = String.format("name = '%s'", modelName);
        List<ModelVersion> mvs1 = client.searchModelVersions(filter1).getItems();
        Assert.assertEquals(mvs1.size(), 2);
        Assert.assertEquals(mvs1.get(0).getName(), modelName);
        Assert.assertEquals(mvs1.get(1).getName(), modelName);

        String filter2 = String.format("name = '%s'", modelName2);
        List<ModelVersion> mvs2 = client.searchModelVersions(filter2).getItems();
        Assert.assertEquals(mvs2.size(), 1);
        Assert.assertEquals(mvs2.get(0).getName(), modelName2);
        Assert.assertEquals(mvs2.get(0).getVersion(), "1");

        String filter3 = String.format("run_id = '%s'", newVersionRunId);
        List<ModelVersion> mvs3 = client.searchModelVersions(filter3).getItems();
        Assert.assertEquals(mvs3.size(), 1);
        Assert.assertEquals(mvs3.get(0).getName(), modelName);
        Assert.assertEquals(mvs3.get(0).getVersion(), "2");

        ModelVersionsPage page1 = client.searchModelVersions(
            "", 1, Arrays.asList("creation_timestamp ASC")
        );
        Assert.assertEquals(page1.getItems().size(), 1);
        Assert.assertEquals(page1.getItems().get(0).getName(), modelName);
        Assert.assertTrue(page1.getNextPageToken().isPresent());

        ModelVersionsPage page2 = client.searchModelVersions(
            "",
            2,
            Arrays.asList("creation_timestamp ASC"),
            page1.getNextPageToken().get()
        );
        Assert.assertEquals(page2.getItems().size(), 2);
        Assert.assertEquals(page2.getItems().get(0).getName(), modelName);
        Assert.assertEquals(page2.getItems().get(0).getRunId(), newVersionRunId);
        Assert.assertEquals(page2.getItems().get(1).getName(), modelName2);
        Assert.assertEquals(page2.getItems().get(1).getRunId(), runId2);
        Assert.assertFalse(page2.getNextPageToken().isPresent());

        ModelVersionsPage nextPageFromPrevPage = (ModelVersionsPage) page1.getNextPage();
        Assert.assertEquals(nextPageFromPrevPage.getItems().size(), 1);
        Assert.assertEquals(page2.getItems().get(0).getName(), modelName);
        Assert.assertEquals(page2.getItems().get(0).getRunId(), newVersionRunId);
        Assert.assertTrue(nextPageFromPrevPage.getNextPageToken().isPresent());
    }
}
