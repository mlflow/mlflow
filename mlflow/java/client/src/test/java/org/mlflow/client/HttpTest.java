package org.mlflow.client;

import org.testng.annotations.*;

import java.io.IOException;

public class HttpTest {
  private final TestClientProvider testClientProvider = new TestClientProvider();

  private ApiClient client;

  @BeforeSuite
  public void beforeAll() throws IOException {
    client = testClientProvider.initializeClientAndServer();
  }

  @AfterSuite
  public void afterAll() throws InterruptedException {
    testClientProvider.cleanupClientAndServer();
  }

  @Test(expectedExceptions = HttpClientException.class)
  public void nonExistentPath() throws Exception {
    client.get("BAD_PATH");
  }

  @Test(expectedExceptions = HttpServerException.class)   // TODO: server should throw 4xx
  public void getExperiment_NonExistentId() throws Exception {
    client.get("experiments/get?experiment_id=NON_EXISTENT_EXPERIMENT_ID");
  }

  @Test(expectedExceptions = HttpServerException.class) // TODO: server should throw 4xx
  public void createExperiment_IllegalJsonSyntax() throws Exception {
    client.post("experiments/create", "NOT_JSON");
  }

  @Test(expectedExceptions = HttpServerException.class) // TODO: server should throw 4xx
  public void createExperiment_MissingJsonField() throws Exception {
    String data = "{\"BAD_name\": \"EXPERIMENT_NAME\"}";
    client.post("experiments/create", data);
  }
}
