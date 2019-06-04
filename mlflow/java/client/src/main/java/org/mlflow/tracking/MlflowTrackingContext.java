package org.mlflow.tracking;

import org.mlflow.api.proto.Service.*;

import java.util.ArrayDeque;
import java.util.Deque;

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
                .build();
        RunInfo runInfo = client.createRun(createRunRequest);
        ActiveRun activeRun = new ActiveRun(runInfo, client);
        activeRunStack.push(activeRun);
        return activeRun;
    }

    public ActiveRun endRun() {
        ActiveRun endedRun = activeRunStack.pop();
        client.setTerminated(endedRun.runInfo.getRunId(), RunStatus.FINISHED);
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
