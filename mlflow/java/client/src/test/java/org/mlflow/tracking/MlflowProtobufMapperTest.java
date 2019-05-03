package org.mlflow.tracking;

import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.testng.Assert;
import org.testng.annotations.Test;

import org.mlflow.api.proto.Service;

public class MlflowProtobufMapperTest {
  @Test
  public void testSerializeSnakeCase() {
    MlflowProtobufMapper mapper = new MlflowProtobufMapper();
    String result = mapper.makeLogParam("my-id", "my-key", "my-value");

    Gson gson = new Gson();
    Type type = new TypeToken<Map<String, Object>>(){}.getType();
    Map<String, String> serializedMessage = gson.fromJson(result, type);

    Map<String, String> expectedMessage = new HashMap<>();
    expectedMessage.put("run_uuid", "my-id");
    expectedMessage.put("run_id", "my-id");
    expectedMessage.put("key", "my-key");
    expectedMessage.put("value", "my-value");
    Assert.assertEquals(serializedMessage, expectedMessage);
  }

  @Test
  public void testDeserializeSnakeCaseAndUnknown() {
    MlflowProtobufMapper mapper = new MlflowProtobufMapper();
    Service.CreateExperiment.Response result = mapper.toCreateExperimentResponse(
      "{\"experiment_id\": 123, \"what is this field\": \"even\"}");
    Assert.assertEquals(result.getExperimentId(), "123");
  }
}
