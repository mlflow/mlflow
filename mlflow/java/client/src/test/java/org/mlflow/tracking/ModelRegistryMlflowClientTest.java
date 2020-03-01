package org.mlflow.tracking;

import com.google.common.collect.Lists;
import org.apache.commons.io.FileUtils;
import org.mlflow.api.proto.ModelRegistry;
import org.mlflow.api.proto.ModelRegistry.ModelVersion;
import org.mlflow.api.proto.Service.RunInfo;
import org.mockito.Mockito;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.UUID;

import static org.mlflow.tracking.TestUtils.createExperimentName;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;

public class ModelRegistryMlflowClientTest {
    private static final Logger logger = LoggerFactory.getLogger(ModelRegistryMlflowClientTest.class);

    private static final MlflowProtobufMapper mapper = new MlflowProtobufMapper();

    private final TestClientProvider testClientProvider = new TestClientProvider();

    private MlflowClient client;

    private String modelName;
    private File tempFile;

    private static final String content = "Hello, Worldz!";

    @BeforeTest
    public void before() throws IOException {
        client = testClientProvider.initializeClientAndSqlLiteBasedServer();
        modelName = "Model-" + UUID.randomUUID().toString();

        String expName = createExperimentName();
        String expId = client.createExperiment(expName);

        RunInfo runCreated = client.createRun(expId);
        String runId = runCreated.getRunUuid();

        File tempDir = Files.createTempDirectory("tempDir").toFile();
        tempFile = Files.createTempFile(tempDir.toPath(), "file", ".txt").toFile();

        FileUtils.writeStringToFile(tempFile, content, StandardCharsets.UTF_8);

        client.sendPost("registered-models/create",
                mapper.makeCreateModel(modelName));

        client.sendPost("model-versions/create",
                mapper.makeCreateModelVersion(modelName, runId, tempFile.getAbsolutePath()));
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
        // default stages (does not include "None")
        List<ModelVersion> modelVersion = client.getLatestVersions(modelName);
        Assert.assertEquals(modelVersion.size(), 0);
        client.sendPost("model-versions/transition-stage",
                mapper.makeTransitionModelVersionStage(modelName,"1", "Staging"));
        modelVersion = client.getLatestVersions(modelName);
        Assert.assertEquals(modelVersion.size(), 1);
        validateDetailedModelVersion(modelVersion.get(0),
                modelName, "Staging", "1");
    }

    @Test
    public void testGetModelVersionDownloadUri() {
        String downloadUri = client.getModelVersionDownloadUri(modelName, "1");
        Assert.assertEquals(tempFile.getAbsolutePath(), downloadUri);
    }

    @Test
    public void testDownloadModelVersion() throws IOException {
        File tempDownloadFile = client.downloadModelVersion(modelName, "1");
        String downloadedContent = FileUtils.readFileToString(tempDownloadFile,
                StandardCharsets.UTF_8);
        Assert.assertEquals(content, downloadedContent);
    }

    @Test
    public void testDownloadLatestModelVersion() throws IOException {
        File tempDownloadFile = client.downloadLatestModelVersion(modelName, "None");
        String downloadedContent = FileUtils.readFileToString(tempDownloadFile,
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
}
