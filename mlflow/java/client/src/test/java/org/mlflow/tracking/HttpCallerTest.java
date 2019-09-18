package org.mlflow.tracking;


import org.apache.http.HttpEntity;
import org.apache.http.StatusLine;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpUriRequest;
import org.apache.http.impl.client.CloseableHttpClient;
import org.mlflow.tracking.creds.BasicMlflowHostCreds;
import org.mlflow.tracking.creds.MlflowHostCreds;
import org.mlflow.tracking.creds.MlflowHostCredsProvider;
import org.testng.Assert;
import org.testng.annotations.*;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.mockito.Mockito.*;

public class HttpCallerTest {

  static CloseableHttpClient client = mock(CloseableHttpClient.class);

  static CloseableHttpResponse response429 = mock(CloseableHttpResponse.class);
  static CloseableHttpResponse response200 = mock(CloseableHttpResponse.class);
  static StatusLine statusLine200 = mock(StatusLine.class);
  static HttpEntity entity200 = mock(HttpEntity.class);
  static HttpEntity entity429 = mock(HttpEntity.class);
  static String expectedResponseText = "expected response text.";
  static StatusLine statusLine429 = mock(StatusLine.class);

  MlflowHttpCaller caller = new MlflowHttpCaller(new MlflowHostCredsProvider() {
    @Override
    public MlflowHostCreds getHostCreds() {
      return new BasicMlflowHostCreds("http://some/host");
    }

    @Override
    public void refresh() {
      // pass
    }
  }, 4, 1, client);

  @BeforeSuite
  public void beforeAll() throws IOException {
    when(statusLine200.getStatusCode()).thenReturn(200);
    when(statusLine429.getStatusCode()).thenReturn(429);
    when(response200.getStatusLine()).thenReturn(statusLine200);
    when(entity200.getContent())
            .thenReturn(
                    new ByteArrayInputStream(expectedResponseText.getBytes(StandardCharsets.UTF_8)))
            .thenReturn(new ByteArrayInputStream(expectedResponseText.getBytes(StandardCharsets.UTF_8)));
    when(entity429.getContent())
            .thenReturn(
                    new ByteArrayInputStream(expectedResponseText.getBytes(StandardCharsets.UTF_8)))
            .thenReturn(new ByteArrayInputStream(expectedResponseText.getBytes(StandardCharsets.UTF_8)));
    when(response200.getEntity()).thenReturn(entity200);
    when(response429.getStatusLine()).thenReturn(statusLine429);
    when(response429.getEntity()).thenReturn(entity429);
  }

  @Test
  public void testMultipleRetries() throws IOException {
    when(client.execute(any(HttpUriRequest.class)))
            .thenReturn(response429) // sleep for 1 ms
            .thenReturn(response429) // sleep for 2 ms
            .thenReturn(response429) // sleep for 4 - 3 == 1ms
            .thenReturn(response200);// last response before timing out
    Assert.assertEquals(expectedResponseText, caller.get("some/path"));

    when(client.execute(any(HttpUriRequest.class)))
            .thenReturn(response429) // sleep for 1 ms
            .thenReturn(response429) // sleep for 2 ms
            .thenReturn(response429) // sleep for 4 - 3 == 1ms
            .thenReturn(response200);// last response before timing out
    Assert.assertEquals(expectedResponseText, caller.post("some/path", "{\"attr\":\"val\"}"));
  }

  @Test
  public void testMaxRetryIntervalFor429s() throws IOException {
    when(client.execute(any(HttpUriRequest.class)))
            .thenReturn(response429) // sleep for 1 ms
            .thenReturn(response429) // sleep for 2 ms
            .thenReturn(response429) // sleep for 4 - 3 == 1ms
            .thenReturn(response429) // last response before timing out
            .thenReturn(response200);// should never be returned;
    MlflowHttpException ex = Assert.expectThrows(MlflowHttpException.class,
            () -> caller.get("some/path"));
    Assert.assertEquals(429, ex.getStatusCode());
    when(client.execute(any(HttpUriRequest.class)))
            .thenReturn(response429) // sleep for 1 ms
            .thenReturn(response429) // sleep for 2 ms
            .thenReturn(response429) // sleep for 4 - 3 == 1ms
            .thenReturn(response429) // last response before timing out
            .thenReturn(response200);// should never be returned;
    ex = Assert.expectThrows(MlflowHttpException.class,
            () -> caller.post("some/path", "{\"attr\":\"val\"}"));
    Assert.assertEquals(429, ex.getStatusCode());
  }
}
