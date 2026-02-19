package org.mlflow.tracking;

import org.testng.annotations.*;

import java.io.IOException;

public class HttpTest {
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

  @Test(expectedExceptions = MlflowHttpException.class)
  public void nonExistentPath() {
    client.sendGet("BAD_PATH");
  }

  @Test(expectedExceptions = MlflowHttpException.class)   // TODO: server should throw 4xx
  public void getExperiment_NonExistentId() {
    client.sendGet("experiments/get?experiment_id=NON_EXISTENT_EXPERIMENT_ID");
  }

  @Test(expectedExceptions = MlflowHttpException.class) // TODO: server should throw 4xx
  public void createExperiment_IllegalJsonSyntax() {
    client.sendPost("experiments/create", "NOT_JSON");
  }

  @Test(expectedExceptions = MlflowHttpException.class) // TODO: server should throw 4xx
  public void createExperiment_MissingJsonField() {
    String data = "{\"BAD_name\": \"EXPERIMENT_NAME\"}";
    client.sendPost("experiments/create", data);
  }
}
