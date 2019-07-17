package org.mlflow.tracking;

import java.util.*;

import org.testng.Assert;

import org.mlflow.api.proto.Service.*;

public class TestUtils {

  final static double EPSILON = 0.0001F;

  static boolean equals(double a, double b) {
    return a == b ? true : Math.abs(a - b) < EPSILON;
  }

  static void assertRunInfo(RunInfo runInfo, String experimentId) {
    Assert.assertEquals(runInfo.getExperimentId(), experimentId);
    Assert.assertNotEquals(runInfo.getUserId(), "");
    Assert.assertTrue(runInfo.getStartTime() < runInfo.getEndTime());
  }

  public static void assertParam(List<Param> params, String key, String value) {
    Assert.assertTrue(params.stream().filter(e -> e.getKey().equals(key) && e.getValue().equals(value)).findFirst().isPresent());
  }

  public static void assertMetric(List<Metric> metrics, String key, double value) {

    if (Double.isNaN(value)) {
      Assert.assertTrue(metrics.stream().filter(e -> e.getKey().equals(key) && Double.isNaN(e.getValue())).findFirst().isPresent());
    } else if(Double.isInfinite(value) && value > 0) {
      Assert.assertTrue(metrics.stream().filter(e -> e.getKey().equals(key) && e.getValue() >= Double.MAX_VALUE).findFirst().isPresent());
    } else if(Double.isInfinite(value) && value < 0) {
      Assert.assertTrue(metrics.stream().filter(e -> e.getKey().equals(key) && e.getValue() <= -Double.MAX_VALUE).findFirst().isPresent());
    } else {
      Assert.assertTrue(metrics.stream().filter(e -> e.getKey().equals(key) && equals(e.getValue(), value)).findFirst().isPresent());
    }
  }

  public static void assertMetric(List<Metric> metrics, String key, double value, long timestamp, long step) {
    Assert.assertTrue(metrics.stream().filter(
      e -> e.getKey().equals(key) && equals(e.getValue(), value) && equals(e.getTimestamp(), timestamp)
      && equals(e.getStep(), step)).findFirst().isPresent());
  }

  public static void assertMetricHistory(List<Metric> history, String key, List<Double> values, List<Long> steps) {
    Assert.assertEquals(history.size(), values.size());
    Assert.assertEquals(history.size(), steps.size());
    for (int i = 0; i < history.size(); i++) {
      Metric metric = history.get(i);
      Assert.assertEquals(metric.getKey(), key);
      Assert.assertTrue(equals(metric.getValue(), values.get(i)));
      Assert.assertTrue(equals(metric.getStep(), steps.get(i)));
    }
  }

  public static void assertMetricHistory(List<Metric> history, String key, List<Double> values, List<Long> timestamps, List<Long> steps) {
    assertMetricHistory(history, key, values, steps);
    for(int i = 0; i < history.size(); ++i) {
      Assert.assertTrue(equals(history.get(i).getTimestamp(), timestamps.get(i)));
    }
  }

  public static void assertTag(List<RunTag> tags, String key, String value) {
    Assert.assertTrue(tags.stream().filter(e -> e.getKey().equals(key) && e.getValue().equals(value)).findFirst().isPresent());
  }
  public static java.util.Optional<Experiment> getExperimentByName(List<Experiment> exps, String expName) {
    return exps.stream().filter(e -> e.getName().equals(expName)).findFirst();
  }

  static public String createExperimentName() {
    return "JavaTestExp_" + UUID.randomUUID().toString();
  }

  public static Metric createMetric(String name, double value, long timestamp, long step) {
    Metric.Builder builder = Metric.newBuilder();
    builder.setKey(name).setValue(value).setTimestamp(timestamp);
    builder.setKey(name).setValue(value).setStep(step);
    return builder.build();
  }

  public static Param createParam(String name, String value) {
    Param.Builder builder = Param.newBuilder();
    builder.setKey(name).setValue(value);
    return builder.build();
  }

  public static RunTag createTag(String name, String value) {
    RunTag.Builder builder = RunTag.newBuilder();
    builder.setKey(name).setValue(value);
    return builder.build();
  }
}
