package org.mlflow.tracking;

import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.utils.DatabricksContext;
import org.mlflow.tracking.utils.MlflowTagConstants;

import java.util.*;
import java.util.function.Consumer;

public class MlflowTrackingContext {
  private MlflowClient client;
  private String experimentId;
  private Deque<ActiveRun> activeRunStack = new ArrayDeque<>();

  public MlflowTrackingContext() {
    this(new MlflowClient(), getDefaultExperimentId());
  }

  public MlflowTrackingContext(MlflowClient client) {
    this(client, getDefaultExperimentId());
  }

  public MlflowTrackingContext(MlflowClient client, String experimentId) {
    this.client = client;
    this.experimentId = experimentId;
  }

  public MlflowTrackingContext(String experimentId) {
    this(new MlflowClient(), experimentId);
  }

  public ActiveRun startRun(String runName) {
    return startRun(runName, false);
  }

  public ActiveRun startRun(String runName, boolean nested) {
    if (!nested && !activeRunStack.isEmpty()) {
      String existingRunId = getActiveRun().get().getId();
      throw new IllegalArgumentException(String.format("Run with ID %s is already active. To start a nested run call " +
        "startRun with nested=true", existingRunId));
    }
    Map<String, String> tags = new HashMap<>();
    tags.put(MlflowTagConstants.RUN_NAME, runName);
    tags.put(MlflowTagConstants.USER, System.getProperty("user.name"));
    tags.put(MlflowTagConstants.SOURCE_TYPE, "LOCAL");

    if (!activeRunStack.isEmpty()) {
      tags.put(MlflowTagConstants.PARENT_RUN_ID, getActiveRun().get().getId());
    }

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

    ActiveRun activeRun = new ActiveRun(runInfo, client);
    activeRunStack.push(activeRun);
    return activeRun;
  }

  // Context APIs
  public void withActiveRun(String runName, Consumer<ActiveRun> activeRunFunction) {
    ActiveRun newRun = startRun(runName);
    try {
      activeRunFunction.accept(newRun);
    } catch(Exception e) {
      endRun(RunStatus.FAILED);
      return;
    }
    endRun(RunStatus.FINISHED);
  }

  public ActiveRun endRun() {
    return endRun(RunStatus.FINISHED);
  }

  public ActiveRun endRun(RunStatus status) {
    ActiveRun endedRun = activeRunStack.pop();
    client.setTerminated(endedRun.runInfo.getRunId(), status);
    return endedRun;
  }

  /**
   * NULLABLE
   * @return
   */
  public Optional<ActiveRun> getActiveRun() {
    if (activeRunStack.isEmpty()) {
      return Optional.empty();
    }
    return Optional.of(activeRunStack.getFirst());
  }

  private static String getDefaultExperimentId() {
    DatabricksContext databricksContext = DatabricksContext.createIfAvailable();
    if (databricksContext != null) {
      String notebookId = databricksContext.getNotebookId();
      if (notebookId != null) {
        return notebookId;
      }
    }
    return MlflowClient.DEFAULT_EXPERIMENT_ID;
  }
}
