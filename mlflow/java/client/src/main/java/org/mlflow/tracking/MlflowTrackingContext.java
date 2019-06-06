package org.mlflow.tracking;

import org.mlflow.api.proto.Service.*;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.function.Consumer;

public class MlflowTrackingContext {

    private MlflowClient client;
    private String experimentId;
    private Deque<ActiveRun> activeRunStack = new ArrayDeque<>();

    public MlflowTrackingContext() {
        this(new MlflowClient(), MlflowClient.DEFAULT_EXPERIMENT_ID);
    }

    public MlflowTrackingContext(MlflowClient client) {
        this(client, MlflowClient.DEFAULT_EXPERIMENT_ID);
    }

    public MlflowTrackingContext(MlflowClient client, String experimentId) {
        this.client = client;
        this.experimentId = experimentId;
    }

    public MlflowTrackingContext(String experimentId) {
        this(new MlflowClient(), experimentId);
    }

    public static MlflowTrackingContext fromExperimentName(String experimentName) {
        return null;
    }

    public ActiveRun startRun(String runName) {
        CreateRun createRunRequest = CreateRun.newBuilder()
                .setExperimentId(experimentId)
                .setStartTime(System.currentTimeMillis())
                .addTags(RunTag.newBuilder().setKey("mlflow.runName").setValue(runName).build())
                .addTags(RunTag.newBuilder().setKey("mlflow.user")
                        .setValue(System.getProperty("user.name")).build())
                .addTags(RunTag.newBuilder().setKey("mlflow.source.type")
                        .setValue("LOCAL").build())
                .build();
        RunInfo runInfo = client.createRun(createRunRequest);
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
    public ActiveRun getActiveRun() {
        return activeRunStack.getFirst();
    }
}
