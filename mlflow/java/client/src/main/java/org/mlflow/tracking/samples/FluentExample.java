package org.mlflow.tracking.samples;

import com.google.common.collect.ImmutableMap;
import org.mlflow.tracking.ActiveRun;
import org.mlflow.tracking.MlflowContext;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FluentExample {
  public static void main(String[] args) {
    MlflowContext mlflow = new MlflowContext();
    ExecutorService executor = Executors.newFixedThreadPool(10);

    // Vanilla usage
    {
      ActiveRun run = mlflow.startRun("run");
      run.logParam("alpha", "0.0");
      run.logMetric("MSE", 0.0);
      run.setTags(ImmutableMap.of(
        "company", "databricks",
        "org", "engineering"
      ));
      run.endRun();
    }

    // Lambda usage
    {
      mlflow.withActiveRun("lambda run", (activeRun -> {
        activeRun.logParam("layers", "4");
        // Perform training code
      }));
    }
    // Log one parent run and 5 children run
    {
      ActiveRun run = mlflow.startRun("parent run");
      for (int i = 0; i <= 5; i++) {
        ActiveRun childRun = mlflow.startRun("child run", run.getId());
        childRun.logParam("iteration", Integer.toString(i));
        childRun.endRun();
      }
      run.endRun();
    }

    // Log one parent run and 5 children run (multithreaded)
    {
      ActiveRun run = mlflow.startRun("parent run (multithreaded)");
      for (int i = 0; i <= 5; i++) {
        final int i0 = i;
        executor.submit(() -> {
          ActiveRun childRun = mlflow.startRun("child run (multithreaded)", run.getId());
          childRun.logParam("iteration", Integer.toString(i0));
          childRun.endRun();
        });
      }
      run.endRun();
    }
    executor.shutdown();
    mlflow.getClient().close();
  }
}
