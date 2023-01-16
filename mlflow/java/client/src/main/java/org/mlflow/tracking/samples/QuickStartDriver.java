package org.mlflow.tracking.samples;

import java.util.List;
import java.util.Optional;

import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.MlflowClient;

/**
 * This is an example application which uses the MLflow Tracking API to create and manage
 * experiments and runs.
 */
public class QuickStartDriver {
  public static void main(String[] args) throws Exception {
    (new QuickStartDriver()).process(args);
  }

  void process(String[] args) throws Exception {
    MlflowClient client;
    if (args.length < 1) {
      client = new MlflowClient();
    } else {
      client = new MlflowClient(args[0]);
    }

    System.out.println("====== createExperiment");
    String expName = "Exp_" + System.currentTimeMillis();
    String expId = client.createExperiment(expName);
    System.out.println("createExperiment: expId=" + expId);

    System.out.println("====== getExperiment");
    Experiment exp = client.getExperiment(expId);
    System.out.println("getExperiment: " + exp);

    System.out.println("====== searchExperiments");
    List<Experiment> exps = client.searchExperiments().getItems();
    System.out.println("#experiments: " + exps.size());
    exps.forEach(e -> System.out.println("  Exp: " + e));

    createRun(client, expId);

    System.out.println("====== getExperiment again");
    Experiment exp2 = client.getExperiment(expId);
    System.out.println("getExperiment: " + exp2);

    System.out.println("====== getExperiment by name");
    Optional<Experiment> exp3 = client.getExperimentByName(expName);
    System.out.println("getExperimentByName: " + exp3);
    client.close();
  }

  void createRun(MlflowClient client, String expId) {
    System.out.println("====== createRun");

    // Create run
    String sourceFile = "MyFile.java";

    RunInfo runCreated = client.createRun(expId);
    System.out.println("CreateRun: " + runCreated);
    String runId = runCreated.getRunUuid();

    // Log parameters
    client.logParam(runId, "min_samples_leaf", "2");
    client.logParam(runId, "max_depth", "3");

    // Log metrics
    client.logMetric(runId, "auc", 2.12F);
    client.logMetric(runId, "accuracy_score", 3.12F);
    client.logMetric(runId, "zero_one_loss", 4.12F);

    // Update finished run
    client.setTerminated(runId, RunStatus.FINISHED);

    // Get run details
    Run run = client.getRun(runId);
    System.out.println("GetRun: " + run);
  }
}
