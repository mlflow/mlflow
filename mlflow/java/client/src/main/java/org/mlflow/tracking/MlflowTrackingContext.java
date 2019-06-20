package org.mlflow.tracking;

import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.utils.DatabricksContext;
import org.mlflow.tracking.utils.MlflowTagConstants;

import java.util.*;
import java.util.function.Consumer;

public class MlflowTrackingContext {
  private MlflowClient client;
  private String experimentId;

  public MlflowTrackingContext() {
    this(new MlflowClient(), getDefaultExperimentId());
  }

  public MlflowTrackingContext(MlflowClient client) {
    this(client, getDefaultExperimentId());
  }

  public MlflowTrackingContext(String experimentId) {
    this(new MlflowClient(), experimentId);
  }

  public MlflowTrackingContext(MlflowClient client, String experimentId) {
    this.client = client;
    this.experimentId = experimentId;
  }

  public ActiveRun startRun(String runName) {
    Map<String, String> tags = new HashMap<>();
    tags.put(MlflowTagConstants.RUN_NAME, runName);
    tags.put(MlflowTagConstants.USER, System.getProperty("user.name"));
    tags.put(MlflowTagConstants.SOURCE_TYPE, "LOCAL");


    // Add tags from DatabricksContext if they exist
    DatabricksContext databricksContext = DatabricksContext.createIfAvailable();
    if (databricksContext != null) {
      tags.putAll(databricksContext.getTags());
    }

    CreateRun.Builder createRunBuilder = CreateRun.newBuilder()
      .setExperimentId(experimentId)
      .setStartTime(System.currentTimeMillis());
    for (Map.Entry<String, String> tag: tags.entrySet()) {
      createRunBuilder.addTags(
        RunTag.newBuilder().setKey(tag.getKey()).setValue(tag.getValue()).build());
    }
    RunInfo runInfo = client.createRun(createRunBuilder.build());

    return new ActiveRun(runInfo, client, experimentId);
  }

  // Context APIs
  public void withActiveRun(String runName, Consumer<ActiveRun> activeRunFunction) {
    ActiveRun newRun = startRun(runName);
    try {
      activeRunFunction.accept(newRun);
    } catch(Exception e) {
      newRun.endRun(RunStatus.FAILED);
      return;
    }
    newRun.endRun(RunStatus.FINISHED);
  }

  private static String getDefaultExperimentId() {
    DatabricksContext databricksContext = DatabricksContext.createIfAvailable();
    if (databricksContext != null && databricksContext.isInDatabricksNotebook()) {
      String notebookId = databricksContext.getNotebookId();
      if (notebookId != null) {
        return notebookId;
      }
    }
    return MlflowClient.DEFAULT_EXPERIMENT_ID;
  }
}
