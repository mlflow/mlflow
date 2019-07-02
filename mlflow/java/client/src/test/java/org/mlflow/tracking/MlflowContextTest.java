package org.mlflow.tracking;

import static org.mockito.Mockito.*;

import static org.mlflow.api.proto.Service.*;

import org.mlflow.tracking.utils.MlflowTagConstants;
import org.mockito.ArgumentCaptor;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MlflowContextTest {
  private static MlflowClient mockClient;
  private static ExecutorService executor;

  @BeforeSuite
  public static void beforeAll() {
    executor = Executors.newFixedThreadPool(5);
  }

  @AfterMethod
  public static void afterMethod() {
    mockClient = null;
  }

  public static MlflowContext setupMlflowContext() {
    MlflowContext mlflow = MlflowContext.getOrCreate("http://localhost:5000");
    mockClient = mock(MlflowClient.class);
    mlflow.setClient(mockClient);
    return mlflow;
  }

  @Test
  public void testGetOrCreate() {
    MlflowContext.defaultContext = null;
    MlflowContext a = MlflowContext.getOrCreate("http://localhost:5000");
    MlflowContext b = MlflowContext.getOrCreate("http://localhost:5001");
    Assert.assertEquals(a, b);
    Assert.assertEquals(a.getClient().getInternalHostCredsProvider().getHostCreds().getHost(),
      "http://localhost:5000");
  }

  @Test
  public void testActiveRun() {
    // Will return !.isPresent() if active run is not created yet.
    {
      MlflowContext mlflow = setupMlflowContext();
      Assert.assertFalse(mlflow.activeRun().isPresent());
    }

    // Will return last activeRun created.
    {
      MlflowContext mlflow = setupMlflowContext();
      when(mockClient.createRun(any(CreateRun.class))).thenReturn(
        RunInfo.newBuilder().setRunId("1").build());
      mlflow.startRun("test-1");
      when(mockClient.createRun(any(CreateRun.class))).thenReturn(
        RunInfo.newBuilder().setRunId("2").build());
      mlflow.startRun("test-2");
      Assert.assertTrue(mlflow.activeRun().isPresent());
      Assert.assertEquals(mlflow.activeRun().get().getId(), "2");
    }

    // If test-1 and test-2 are started and then terminated, then startRun will return !isPresent option.
    {
      MlflowContext mlflow = setupMlflowContext();
      when(mockClient.createRun(any(CreateRun.class))).thenReturn(
        RunInfo.newBuilder().setRunId("1").build());
      ActiveRun test1 = mlflow.startRun("test-1");
      when(mockClient.createRun(any(CreateRun.class))).thenReturn(
        RunInfo.newBuilder().setRunId("2").build());
      ActiveRun test2 = mlflow.startRun("test-2");

      // It actually doesn't matter the order in which the runs are ended.
      test1.endRun();
      test2.endRun();

      Assert.assertTrue(mlflow.activeRun().isPresent());
      Assert.assertEquals(mlflow.activeRun().get().getId(), "2");
    }
  }

  @Test
  public void testSetExperimentName() {
    // Will throw if there is no experiment with the same name.
    {
      when(mockClient.getExperimentByName("experiment-name")).thenReturn(Optional.empty());
      MlflowContext mlflow = setupMlflowContext();
      try {
        mlflow.setExperimentName("experiment-name");
        Assert.fail();
      } catch (IllegalArgumentException expected) {
      }
    }

    // Will set experiment-id if experiment is returned from getExperimentByName
    {
      when(mockClient.getExperimentByName("experiment-name")).thenReturn(
        Optional.of(Experiment.newBuilder().setExperimentId("123").build()));
      MlflowContext mlflow = setupMlflowContext();
      mlflow.setExperimentName("experiment-name");
      Assert.assertEquals(mlflow.getExperimentId(), "123");
    }
  }

  @Test
  public void testStartRun() {
    // Sets the appropriate tags
    ArgumentCaptor<CreateRun> createRunArgument = ArgumentCaptor.forClass(CreateRun.class);
    MlflowContext mlflow = setupMlflowContext();
    mlflow.setExperimentId("123");
    mlflow.startRun("apple");
    verify(mockClient).createRun(createRunArgument.capture());
    List<RunTag> tags = createRunArgument.getValue().getTagsList();
    Assert.assertEquals(createRunArgument.getValue().getExperimentId(), "123");
    Assert.assertTrue(tags.contains(RunTag.newBuilder().setKey(MlflowTagConstants.RUN_NAME).setValue("apple").build()));
    Assert.assertTrue(tags.contains(createRunTag(MlflowTagConstants.RUN_NAME, "apple")));
    Assert.assertTrue(tags.contains(createRunTag(MlflowTagConstants.SOURCE_TYPE, "LOCAL")));
    Assert.assertTrue(tags.contains(createRunTag(MlflowTagConstants.USER, System.getProperty("user.name"))));
  }

  @Test
  public void parentRunIdSetCorrectly() {
    // Create test-1, test-2, end test-2, create test-3. test-2 and test-3's parent should be test-1
    MlflowContext mlflow = setupMlflowContext();
    when(mockClient.createRun(any(CreateRun.class))).thenReturn(RunInfo.newBuilder().setRunId("1").build());
    mlflow.startRun("test-1");

    reset(mockClient);
    when(mockClient.createRun(any(CreateRun.class))).thenReturn(RunInfo.newBuilder().setRunId("2").build());
    ArgumentCaptor<CreateRun> createRun2 = ArgumentCaptor.forClass(CreateRun.class);
    ActiveRun test2 = mlflow.startRun("test-2");
    verify(mockClient).createRun(createRun2.capture());

    test2.endRun();

    reset(mockClient);
    when(mockClient.createRun(any(CreateRun.class))).thenReturn(RunInfo.newBuilder().setRunId("3").build());
    ArgumentCaptor<CreateRun> createRun3 = ArgumentCaptor.forClass(CreateRun.class);
    mlflow.startRun("test-3");
    verify(mockClient).createRun(createRun3.capture());

    Assert.assertTrue(createRun2.getValue().getTagsList().contains(createRunTag(MlflowTagConstants.PARENT_RUN_ID, "1")));
    Assert.assertTrue(createRun3.getValue().getTagsList().contains(createRunTag(MlflowTagConstants.PARENT_RUN_ID, "1")));
  }

  @Test
  public void startRunParallel() throws InterruptedException, ExecutionException {
    // Create test-1, test-2, end test-2, create test-3. test-2 and test-3's parent should be test-1
    MlflowContext mlflow = setupMlflowContext();
    ArgumentCaptor<CreateRun> createRun = ArgumentCaptor.forClass(CreateRun.class);
    List<Future<?>> futures = new ArrayList<>();
    for (int i = 0; i < 5; i++) {
      final int i0 = i;
      futures.add(executor.submit(() -> {
        mlflow.startRun(Integer.toString(i0));
      }));
    }

    for (Future<?> f: futures) {
      f.get();
    }

    verify(mockClient, times(5)).createRun(createRun.capture());
    for (CreateRun c: createRun.getAllValues()) {
      Assert.assertFalse(c.getTagsList().contains(createRunTag(MlflowTagConstants.PARENT_RUN_ID, "1")));
    }
  }

  private static RunTag createRunTag(String key, String value) {
    return RunTag.newBuilder().setKey(key).setValue(value).build();
  }
}