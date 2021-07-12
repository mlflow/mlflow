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

  static CloseableHttpResponse response200 = mock(CloseableHttpResponse.class);
  static CloseableHttpResponse response429 = mock(CloseableHttpResponse.class);
  static CloseableHttpResponse response500 = mock(CloseableHttpResponse.class);
  static HttpEntity entity200 = mock(HttpEntity.class);
  static HttpEntity entity429 = mock(HttpEntity.class);
  static HttpEntity entity500 = mock(HttpEntity.class);
  static StatusLine statusLine200 = mock(StatusLine.class);
  static StatusLine statusLine429 = mock(StatusLine.class);
  static StatusLine statusLine500 = mock(StatusLine.class);
  static String expectedResponseText = "expected response text.";

  MlflowHttpCaller caller = new MlflowHttpCaller(new MlflowHostCredsProvider() {
    @Override
    public MlflowHostCreds getHostCreds() {
      return new BasicMlflowHostCreds("http://some/host");
    }

    @Override
    public void refresh() {
      // pass
    }
  }, 4, 1, 3, client);

  @BeforeSuite
  public void beforeAll() throws IOException {
    when(statusLine200.getStatusCode()).thenReturn(200);
    when(response200.getStatusLine()).thenReturn(statusLine200);
    when(entity200.getContent()).thenAnswer(
      i -> new ByteArrayInputStream(expectedResponseText.getBytes(StandardCharsets.UTF_8))
    );
    when(response200.getEntity()).thenReturn(entity200);
    when(statusLine429.getStatusCode()).thenReturn(429);
    when(response429.getStatusLine()).thenReturn(statusLine429);
    when(entity429.getContent()).thenAnswer(
      i -> new ByteArrayInputStream(expectedResponseText.getBytes(StandardCharsets.UTF_8))
    );
    when(response429.getEntity()).thenReturn(entity429);
    when(statusLine500.getStatusCode()).thenReturn(500);
    when(response500.getStatusLine()).thenReturn(statusLine500);
    when(entity500.getContent()).thenAnswer(
      i -> new ByteArrayInputStream(expectedResponseText.getBytes(StandardCharsets.UTF_8))
    );
    when(response500.getEntity()).thenReturn(entity500);
  }

  @Test
  public void testRequestsAreRetriedFor429s() throws IOException {
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
  public void testMaxRetryIntervalIsRespectedFor429s() throws IOException {
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

  @Test
  public void testRequestsAreRetriedFor500s() throws IOException {
    when(client.execute(any(HttpUriRequest.class)))
            .thenReturn(response500)
            .thenReturn(response429)
            .thenReturn(response429)
            .thenReturn(response500)
            .thenReturn(response429)
            .thenReturn(response429)
            .thenReturn(response200);
    Assert.assertEquals(expectedResponseText, caller.get("some/path"));

    when(client.execute(any(HttpUriRequest.class)))
            .thenReturn(response500)
            .thenReturn(response429)
            .thenReturn(response429)
            .thenReturn(response500)
            .thenReturn(response429)
            .thenReturn(response429)
            .thenReturn(response200);
    Assert.assertEquals(expectedResponseText, caller.post("some/path", "{\"attr\":\"val\"}"));
  }

  @Test
  public void testMaxRetryAttemptsIsRespectedFor500s() throws IOException {
    when(client.execute(any(HttpUriRequest.class)))
            .thenReturn(response500)
            .thenReturn(response500)
            .thenReturn(response500) // last response before timing out
            .thenReturn(response200);// should never be returned;
    MlflowHttpException ex = Assert.expectThrows(MlflowHttpException.class,
            () -> caller.get("some/path"));
    Assert.assertEquals(500, ex.getStatusCode());

    when(client.execute(any(HttpUriRequest.class)))
            .thenReturn(response500)
            .thenReturn(response500)
            .thenReturn(response500) // last response before timing out
            .thenReturn(response200);// should never be returned;
    ex = Assert.expectThrows(MlflowHttpException.class,
            () -> caller.post("some/path", "{\"attr\":\"val\"}"));
    Assert.assertEquals(500, ex.getStatusCode());
  }
}
