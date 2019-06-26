package org.mlflow.tracking;

import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.utils.DatabricksContext;
import org.mlflow.tracking.utils.MlflowTagConstants;

import java.util.*;
import java.util.function.Consumer;

public class MlflowContext {
  private static MlflowContext defaultContext;
  private MlflowClient client;
  private String experimentId;
  private ThreadLocal<Deque<ActiveRun>> perThreadActiveRunStack =
    ThreadLocal.withInitial(ArrayDeque::new);

  public MlflowContext() {
    this(new MlflowClient(), getDefaultExperimentId());
  }

  public MlflowContext(MlflowClient client) {
    this(client, getDefaultExperimentId());
  }

  public MlflowContext(String experimentId) {
    this(new MlflowClient(), experimentId);
  }

  public MlflowContext(MlflowClient client, String experimentId) {
    this.client = client;
    this.experimentId = experimentId;
  }

  public static synchronized MlflowContext getOrCreate() {
    return getOrCreate(new MlflowClient(), getDefaultExperimentId());
  }

  public static synchronized MlflowContext getOrCreate(String trackingUri) {
    return getOrCreate(trackingUri, getDefaultExperimentId());
  }

  public static synchronized MlflowContext getOrCreate(String trackingUri, String experimentId) {
    return getOrCreate(new MlflowClient(trackingUri), experimentId);
  }

  private static synchronized MlflowContext getOrCreate(MlflowClient client, String experimentId) {
    if (defaultContext != null) {
      return defaultContext;
    }
    defaultContext = new MlflowContext(client, experimentId);
    return defaultContext;
  }

  public synchronized Optional<ActiveRun> activeRun() {
    cleanupActiveRunStack();
    return Optional.ofNullable(perThreadActiveRunStack.get().peekFirst());
  }

  public void setClient(MlflowClient client) {
    this.client = client;
  }

  public void setExperimentName(String experimentName) {
    Optional<Experiment> experimentOpt = client.getExperimentByName(experimentName);
    if (!experimentOpt.isPresent()) {
      throw new IllegalArgumentException(
        String.format("%s is not a valid experiment", experimentName));
    }
    experimentId = experimentOpt.get().getExperimentId();
  }

  public void setExperimentId(String experimentId) {
    this.experimentId = experimentId;
  }

  public ActiveRun startRun(String runName) {
    return startRun(runName, null);
  }

  public ActiveRun startRun(String runName, String parentRunId) {
    cleanupActiveRunStack();
    Map<String, String> tags = new HashMap<>();
    tags.put(MlflowTagConstants.RUN_NAME, runName);
    tags.put(MlflowTagConstants.USER, System.getProperty("user.name"));
    tags.put(MlflowTagConstants.SOURCE_TYPE, "LOCAL");
    if (parentRunId != null) {
      tags.put(MlflowTagConstants.PARENT_RUN_ID, parentRunId);
    } else if (activeRun().isPresent()){
      tags.put(MlflowTagConstants.PARENT_RUN_ID, activeRun().get().getId());
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

    ActiveRun newRun = new ActiveRun(runInfo, client);
    perThreadActiveRunStack.get().push(newRun);
    return newRun;
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

  private void cleanupActiveRunStack() {
    Deque<ActiveRun> activeRunStack = perThreadActiveRunStack.get();
    ActiveRun lastActiveRun = activeRunStack.peekFirst();
    while (lastActiveRun != null && lastActiveRun.isTerminated) {
      activeRunStack.pop();
      lastActiveRun = activeRunStack.peekFirst();
    }
  }
}
