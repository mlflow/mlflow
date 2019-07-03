package org.mlflow.tracking;

import com.google.common.collect.ImmutableMap;
import org.mlflow.api.proto.Service.*;
import org.mockito.ArgumentCaptor;
import org.testng.annotations.Test;
import org.testng.collections.Lists;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.testng.Assert.*;
import static org.mockito.Mockito.*;

public class ActiveRunTest {

  private MlflowClient mockClient;

  private ActiveRun getActiveRun() {
    RunInfo r = RunInfo.newBuilder().setRunId("test-run-id").build();
    this.mockClient = mock(MlflowClient.class);
    return new ActiveRun(r, mockClient);
  }

  @Test
  public void testLogMetrics() {
    ActiveRun activeRun = getActiveRun();
    ArgumentCaptor<Iterable<Metric>> metricsArg = ArgumentCaptor.forClass(Iterable.class);
    activeRun.logMetrics(ImmutableMap.of("a", 0.0, "b", 1.0));
    verify(mockClient).logBatch(eq("test-run-id"), metricsArg.capture(), any(), any());

    Set<Metric> metrics = new HashSet<>();
    metricsArg.getValue().forEach(metrics::add);

    assertTrue(metrics.stream().anyMatch(m -> m.getKey().equals("a") && m.getValue() == 0.0));
    assertTrue(metrics.stream().anyMatch(m -> m.getKey().equals("b") && m.getValue() == 1.0));
  }
}