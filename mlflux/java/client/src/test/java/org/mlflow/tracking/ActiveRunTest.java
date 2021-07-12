package org.mlflow.tracking;

import com.google.common.collect.ImmutableMap;
import org.mlflow.api.proto.Service.*;
import org.mockito.ArgumentCaptor;
import org.testng.annotations.Test;
import org.testng.collections.Lists;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.testng.Assert.*;
import static org.mockito.Mockito.*;

public class ActiveRunTest {

  private static final String RUN_ID = "test-run-id";
  private static final String ARTIFACT_URI = "dbfs:/artifact-uri";

  private MlflowClient mockClient;

  private ActiveRun getActiveRun() {
    RunInfo r = RunInfo.newBuilder().setRunId(RUN_ID).setArtifactUri(ARTIFACT_URI).build();
    this.mockClient = mock(MlflowClient.class);
    return new ActiveRun(r, mockClient);
  }

  @Test
  public void testGetId() {
    assertEquals(getActiveRun().getId(), RUN_ID);
  }

  @Test
  public void testLogParam() {
    getActiveRun().logParam("param-key", "param-value");
    verify(mockClient).logParam(RUN_ID, "param-key", "param-value");
  }

  @Test
  public void testSetTag() {
    getActiveRun().setTag("tag-key", "tag-value");
    verify(mockClient).setTag(RUN_ID, "tag-key", "tag-value");
  }

  @Test
  public void testLogMetric() {
    getActiveRun().logMetric("metric-key", 1.0);
    // The any is for the timestamp.
    verify(mockClient).logMetric(eq(RUN_ID), eq("metric-key"), eq(1.0), anyLong(), eq(0L));
  }

  @Test
  public void testLogMetricWithStep() {
    getActiveRun().logMetric("metric-key", 1.0, 99);
    // The any is for the timestamp.
    verify(mockClient).logMetric(eq(RUN_ID), eq("metric-key"), eq(1.0), anyLong(), eq(99L));
  }

  @Test
  public void testLogMetrics() {
    ActiveRun activeRun = getActiveRun();
    ArgumentCaptor<Iterable<Metric>> metricsArg = ArgumentCaptor.forClass(Iterable.class);
    activeRun.logMetrics(ImmutableMap.of("a", 0.0, "b", 1.0));
    verify(mockClient).logBatch(eq(RUN_ID), metricsArg.capture(), any(), any());

    Set<Metric> metrics = new HashSet<>();
    metricsArg.getValue().forEach(metrics::add);

    assertTrue(metrics.stream()
      .anyMatch(m -> m.getKey().equals("a") && m.getValue() == 0.0 && m.getStep() == 0));
    assertTrue(metrics.stream()
      .anyMatch(m -> m.getKey().equals("b") && m.getValue() == 1.0 && m.getStep() == 0));
  }

  @Test
  public void testLogMetricsWithStep() {
    ActiveRun activeRun = getActiveRun();
    ArgumentCaptor<Iterable<Metric>> metricsArg = ArgumentCaptor.forClass(Iterable.class);
    activeRun.logMetrics(ImmutableMap.of("a", 0.0, "b", 1.0), 99);
    verify(mockClient).logBatch(eq(RUN_ID), metricsArg.capture(), any(), any());

    Set<Metric> metrics = new HashSet<>();
    metricsArg.getValue().forEach(metrics::add);

    assertTrue(metrics.stream()
      .anyMatch(m -> m.getKey().equals("a") && m.getValue() == 0.0 && m.getStep() == 99));
    assertTrue(metrics.stream()
      .anyMatch(m -> m.getKey().equals("b") && m.getValue() == 1.0 && m.getStep() == 99));
  }

  @Test
  public void testLogParams() {
    ActiveRun activeRun = getActiveRun();
    ArgumentCaptor<Iterable<Param>> paramsArg = ArgumentCaptor.forClass(Iterable.class);
    activeRun.logParams(ImmutableMap.of("a", "a", "b", "b"));
    verify(mockClient).logBatch(eq(RUN_ID), any(), paramsArg.capture(), any());

    Set<Param> params = new HashSet<>();
    paramsArg.getValue().forEach(params::add);

    assertTrue(params.stream()
      .anyMatch(p -> p.getKey().equals("a") && p.getValue().equals("a")));
    assertTrue(params.stream()
      .anyMatch(p -> p.getKey().equals("b") && p.getValue().equals("b")));
  }

  @Test
  public void testSetTags() {
    ActiveRun activeRun = getActiveRun();
    ArgumentCaptor<Iterable<RunTag>> tagsArg = ArgumentCaptor.forClass(Iterable.class);
    activeRun.setTags(ImmutableMap.of("a", "a", "b", "b"));
    verify(mockClient).logBatch(eq(RUN_ID), any(), any(), tagsArg.capture());

    Set<RunTag> tags = new HashSet<>();
    tagsArg.getValue().forEach(tags::add);

    assertTrue(tags.stream()
      .anyMatch(t -> t.getKey().equals("a") && t.getValue().equals("a")));
    assertTrue(tags.stream()
      .anyMatch(t -> t.getKey().equals("b") && t.getValue().equals("b")));
  }

  @Test
  public void testLogArtifact() {
    ActiveRun activeRun = getActiveRun();
    activeRun.logArtifact(Paths.get("test"));
    verify(mockClient).logArtifact(RUN_ID, new File("test"));
  }

  @Test
  public void testLogArtifactWithArtifactPath() {
    ActiveRun activeRun = getActiveRun();
    activeRun.logArtifact(Paths.get("test"), "artifact-path");
    verify(mockClient).logArtifact(RUN_ID, new File("test"), "artifact-path");
  }

  @Test
  public void testLogArtifacts() {
    ActiveRun activeRun = getActiveRun();
    activeRun.logArtifacts(Paths.get("test"));
    verify(mockClient).logArtifacts(RUN_ID, new File("test"));
  }

  @Test
  public void testLogArtifactsWithArtifactPath() {
    ActiveRun activeRun = getActiveRun();
    activeRun.logArtifacts(Paths.get("test"), "artifact-path");
    verify(mockClient).logArtifacts(RUN_ID, new File("test"), "artifact-path");
  }

  @Test
  public void testGetArtifactUri() {
    ActiveRun activeRun = getActiveRun();
    assertEquals(activeRun.getArtifactUri(), ARTIFACT_URI);
  }

  @Test
  public void testEndRun() {
    ActiveRun activeRun = getActiveRun();
    activeRun.endRun();
    verify(mockClient).setTerminated(RUN_ID, RunStatus.FINISHED);
  }

  @Test
  public void testEndRunWithStatus() {
    ActiveRun activeRun = getActiveRun();
    activeRun.endRun(RunStatus.FAILED);
    verify(mockClient).setTerminated(RUN_ID, RunStatus.FAILED);
  }
}