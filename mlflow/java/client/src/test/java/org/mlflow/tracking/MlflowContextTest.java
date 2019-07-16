package org.mlflow.tracking;

import static org.mockito.Mockito.*;

import static org.mlflow.api.proto.Service.*;

import org.mlflow.tracking.utils.MlflowTagConstants;
import org.mockito.ArgumentCaptor;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.Test;

import java.util.List;
import java.util.Optional;

public class MlflowContextTest {
  private static MlflowClient mockClient;

  @AfterMethod
  public static void afterMethod() {
    mockClient = null;
  }

  public static MlflowContext setupMlflowContext() {
    mockClient = mock(MlflowClient.class);
    MlflowContext mlflow = new MlflowContext(mockClient);
    return mlflow;
  }

  @Test
  public void testGetClient() {
    MlflowContext mlflow = setupMlflowContext();
    Assert.assertEquals(mlflow.getClient(), mockClient);
  }

  @Test
  public void testSetExperimentName() {
    // Will throw if there is no experiment with the same name.
    {
      MlflowContext mlflow = setupMlflowContext();
      when(mockClient.getExperimentByName("experiment-name")).thenReturn(Optional.empty());
      try {
        mlflow.setExperimentName("experiment-name");
        Assert.fail();
      } catch (IllegalArgumentException expected) {
      }
    }

    // Will set experiment-id if experiment is returned from getExperimentByName
    {
      MlflowContext mlflow = setupMlflowContext();
      when(mockClient.getExperimentByName("experiment-name")).thenReturn(
        Optional.of(Experiment.newBuilder().setExperimentId("123").build()));
      mlflow.setExperimentName("experiment-name");
      Assert.assertEquals(mlflow.getExperimentId(), "123");
    }
  }

  @Test
  public void testSetAndGetExperimentId() {
      MlflowContext mlflow = setupMlflowContext();
      mlflow.setExperimentId("apple");
      Assert.assertEquals(mlflow.getExperimentId(), "apple");
  }

  @Test
  public void testStartRun() {
    // Sets the appropriate tags
    ArgumentCaptor<CreateRun> createRunArgument = ArgumentCaptor.forClass(CreateRun.class);
    MlflowContext mlflow = setupMlflowContext();
    mlflow.setExperimentId("123");
    mlflow.startRun("apple", "parent-run-id");
    verify(mockClient).createRun(createRunArgument.capture());
    List<RunTag> tags = createRunArgument.getValue().getTagsList();
    Assert.assertEquals(createRunArgument.getValue().getExperimentId(), "123");
    Assert.assertTrue(tags.contains(createRunTag(MlflowTagConstants.RUN_NAME, "apple")));
    Assert.assertTrue(tags.contains(createRunTag(MlflowTagConstants.SOURCE_TYPE, "LOCAL")));
    Assert.assertTrue(tags.contains(createRunTag(MlflowTagConstants.USER, System.getProperty("user.name"))));
    Assert.assertTrue(tags.contains(createRunTag(MlflowTagConstants.PARENT_RUN_ID, "parent-run-id")));
  }

  @Test
  public void testStartRunWithNoRunName() {
    // Sets the appropriate tags
    ArgumentCaptor<CreateRun> createRunArgument = ArgumentCaptor.forClass(CreateRun.class);
    MlflowContext mlflow = setupMlflowContext();
    mlflow.startRun();
    verify(mockClient).createRun(createRunArgument.capture());
    List<RunTag> tags = createRunArgument.getValue().getTagsList();
    Assert.assertFalse(
      tags.stream().anyMatch(tag -> tag.getKey().equals(MlflowTagConstants.RUN_NAME)));
  }

  @Test
  public void testWithActiveRun() {
    // Sets the appropriate tags
    MlflowContext mlflow = setupMlflowContext();
    mlflow.setExperimentId("123");
    when(mockClient.createRun(any(CreateRun.class)))
      .thenReturn(RunInfo.newBuilder().setRunId("test-id").build());
    mlflow.withActiveRun("apple", activeRun -> {
      Assert.assertEquals(activeRun.getId(), "test-id");
    });
    verify(mockClient).createRun(any(CreateRun.class));
    verify(mockClient).setTerminated(any(), any());
  }

  @Test
  public void testWithActiveRunNoRunName() {
    // Sets the appropriate tags
    MlflowContext mlflow = setupMlflowContext();
    mlflow.setExperimentId("123");
    when(mockClient.createRun(any(CreateRun.class)))
      .thenReturn(RunInfo.newBuilder().setRunId("test-id").build());
    mlflow.withActiveRun(activeRun -> {
      Assert.assertEquals(activeRun.getId(), "test-id");
    });
    verify(mockClient).createRun(any(CreateRun.class));
    verify(mockClient).setTerminated(any(), any());
  }


  private static RunTag createRunTag(String key, String value) {
    return RunTag.newBuilder().setKey(key).setValue(value).build();
  }
}