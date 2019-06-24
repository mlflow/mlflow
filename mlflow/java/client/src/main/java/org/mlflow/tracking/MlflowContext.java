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
  private ActiveRun rootRun;

  public MlflowContext() {
    this(new MlflowClient(), getDefaultExperimentId());
  }

  public MlflowContext(MlflowClient client) {
    this(client, getDefaultExperimentId());
  }

  public MlflowContext(String experimentId) {
    this(new MlflowClient(), experimentId);
  }

  public static synchronized MlflowContext getOrCreate() {
    if (defaultContext != null) {
      return defaultContext;
    }
    defaultContext = new MlflowContext();
    return defaultContext;
  }

  public MlflowContext(MlflowClient client, String experimentId) {
    this.client = client;
    this.experimentId = experimentId;
  }

  public synchronized Optional<ActiveRun> getRootRun() {
    if (rootRun != null && rootRun.isTerminated) {
      rootRun = null;
    }
    return Optional.ofNullable(rootRun);
  }

  public void setClient(MlflowClient client) {
    assertRootRunNotDefined();
    this.client = client;
  }

  public void setExperimentName(String experimentName) {
    assertRootRunNotDefined();
    Optional<Experiment> experimentOpt = client.getExperimentByName(experimentName);
    if (!experimentOpt.isPresent()) {
      throw new IllegalArgumentException(String.format("%s is not a valid experiment", experimentName));
    }
    experimentId = experimentOpt.get().getExperimentId();
  }

  public void setExperimentId(String experimentId) {
    assertRootRunNotDefined();
    this.experimentId = experimentId;
  }

  public synchronized ActiveRun startRun(String runName) {
    if (rootRun != null && !rootRun.isTerminated) {
      throw new IllegalArgumentException("Root run must be terminated before starting a new run");
    }
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

    rootRun = new ActiveRun(runInfo, client, experimentId);
    return rootRun;
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

  private void assertRootRunNotDefined() {
    if (rootRun != null && !rootRun.isTerminated)  {
      throw new IllegalArgumentException("Cannot set new client/experiment if root run is still active");
    }
  }
}
