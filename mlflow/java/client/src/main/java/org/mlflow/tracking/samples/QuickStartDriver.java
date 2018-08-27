package org.mlflow.tracking.samples;

import java.util.*;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;

import org.mlflow.tracking.MlflowClient;
import org.mlflow.api.proto.Service.*;
import org.mlflow.tracking.objects.ObjectUtils;

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

    boolean verbose = args.length >= 2 && "true".equals(args[1]);
    if (verbose) {
      LogManager.getLogger("org.mlflow.client").setLevel(Level.DEBUG);
    }

    System.out.println("====== createExperiment");
    String expName = "Exp_" + System.currentTimeMillis();
    long expId = client.createExperiment(expName);
    System.out.println("createExperiment: expId=" + expId);

    System.out.println("====== getExperiment");
    GetExperiment.Response exp = client.getExperiment(expId);
    System.out.println("getExperiment: " + exp);

    System.out.println("====== listExperiments");
    List<Experiment> exps = client.listExperiments();
    System.out.println("#experiments: " + exps.size());
    exps.forEach(e -> System.out.println("  Exp: " + e));

    createRun(client, expId);

    System.out.println("====== getExperiment again");
    GetExperiment.Response exp2 = client.getExperiment(expId);
    System.out.println("getExperiment: " + exp2);

    System.out.println("====== getExperiment by name");
    Optional<Experiment> exp3 = client.getExperimentByName(expName);
    System.out.println("getExperimentByName: " + exp3);
  }

  void createRun(MlflowClient client, long expId) {
    System.out.println("====== createRun");

    // Create run
    String user = System.getenv("USER");
    long startTime = System.currentTimeMillis();
    String sourceFile = "MyFile.java";

    CreateRun request = ObjectUtils.makeCreateRun(expId, "run_for_" + expId, SourceType.LOCAL,
      sourceFile, startTime, user);
    RunInfo runCreated = client.createRun(request);
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
    client.setTerminated(runId, RunStatus.FINISHED, startTime + 1001);

    // Get run details
    Run run = client.getRun(runId);
    System.out.println("GetRun: " + run);
  }
}
