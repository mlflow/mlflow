package org.mlflow.tracking;

import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.utils.DatabricksContext;
import org.mlflow.tracking.utils.MlflowTagConstants;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

public class MlflowContext {
  private static AtomicBoolean hasInitialized = new AtomicBoolean();

  private static final ThreadLocal<MlflowContext> defaultContexts = new ThreadLocal<MlflowContext>() {
    @Override protected MlflowContext initialValue() {
      // Return a MLflowContext for the first thread which tries to use the defaultContext
      if (hasInitialized.compareAndSet(false, true)) {
        return new MlflowContext();
      }
      // Otherwise return null
      return null;
    }
  };

  private MlflowClient client;
  private String experimentId;
  private Deque<ActiveRun> runStack;

  public MlflowContext copyMlflowContext() {
    // Maybe we should be copying the client here?
    MlflowContext copied = new MlflowContext(client, experimentId);
    copied.runStack = new ArrayDeque<>(this.runStack);
    return copied;
  }

  public static void setDefaultContext(MlflowContext context) {
    defaultContexts.set(context);
  }

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

  public static MlflowContext getDefault() {
    MlflowContext context = defaultContexts.get();
    if (context == null) {
      throw new IllegalArgumentException("Default not set.");
    }
    return context;
  }


  public synchronized Optional<ActiveRun> getActiveRun() {
    return Optional.ofNullable(runStack.peekFirst());
  }

  public void setClient(MlflowClient client) {
    assertStackEmpty();
    this.client = client;
  }

  public void setExperimentName(String experimentName) {
    assertStackEmpty();
    Optional<Experiment> experimentOpt = client.getExperimentByName(experimentName);
    if (!experimentOpt.isPresent()) {
      throw new IllegalArgumentException(String.format("%s is not a valid experiment", experimentName));
    }
    experimentId = experimentOpt.get().getExperimentId();
  }

  public void setExperimentId(String experimentId) {
    assertStackEmpty();
    this.experimentId = experimentId;
  }

  public synchronized ActiveRun startRun(String runName) {
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

    ActiveRun newRun = new ActiveRun(runInfo, client, experimentId);
    runStack.addFirst(newRun);
    return newRun;
  }

  public ActiveRun endRun() {
    return endRun(RunStatus.FINISHED);
  }

  public ActiveRun endRun(RunStatus status) {
    ActiveRun endedRun = runStack.pop();
    client.setTerminated(endedRun.getId(), status);
    return endedRun;
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

  private void assertStackEmpty() {
    if (!runStack.isEmpty())  {
      throw new IllegalArgumentException("Cannot set new client/experiment if there's an active run");
    }
  }
}
