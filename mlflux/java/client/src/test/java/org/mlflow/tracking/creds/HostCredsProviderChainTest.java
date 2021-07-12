package org.mlflow.tracking.creds;

import org.testng.Assert;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import org.mlflow.tracking.MlflowClientException;

public class HostCredsProviderChainTest {
  private boolean refreshCalled = false;
  private boolean throwException = false;
  private MlflowHostCreds providedHostCreds = null;
  private MlflowHostCredsProvider myHostCredsProvider = new MyHostCredsProvider();

  class MyHostCredsProvider implements MlflowHostCredsProvider {
    @Override
    public MlflowHostCreds getHostCreds() {
      if (throwException) {
        throw new IllegalStateException("Oh no!");
      }
      return providedHostCreds;
    }

    @Override
    public void refresh() {
      refreshCalled = true;
    }
  }

  @BeforeMethod
  public void beforeEach() {
    refreshCalled = false;
    throwException = false;
    providedHostCreds = null;
  }

  @Test
  public void testUseFirstIfAvailable() {
    BasicMlflowHostCreds secondCreds = new BasicMlflowHostCreds("hosty", "tokeny");
    HostCredsProviderChain chain = new HostCredsProviderChain(myHostCredsProvider, secondCreds);

    // If we have valid credentials, we should be used.
    providedHostCreds = new BasicMlflowHostCreds("new-host");
    Assert.assertEquals(chain.getHostCreds().getHost(), "new-host");
    Assert.assertNull(chain.getHostCreds().getToken());

    // If our credentials are invalid, we should be skipped.
    providedHostCreds = new BasicMlflowHostCreds(null);
    Assert.assertEquals(chain.getHostCreds().getHost(), "hosty");
    Assert.assertEquals(chain.getHostCreds().getToken(), "tokeny");

    // If we return null, we should be skipped.
    providedHostCreds = null;
    Assert.assertEquals(chain.getHostCreds().getHost(), "hosty");
    Assert.assertEquals(chain.getHostCreds().getToken(), "tokeny");

    // If we return an exception, we should be skipped.
    throwException = true;
    Assert.assertEquals(chain.getHostCreds().getHost(), "hosty");
    Assert.assertEquals(chain.getHostCreds().getToken(), "tokeny");
  }

  @Test
  public void testRefreshPropagates() {
    HostCredsProviderChain chain = new HostCredsProviderChain(myHostCredsProvider);
    Assert.assertFalse(refreshCalled);
    chain.refresh();
    Assert.assertTrue(refreshCalled);
  }

  @Test
  public void testThrowFinalError() {
    throwException = true;
    HostCredsProviderChain chain = new HostCredsProviderChain(myHostCredsProvider);
    try {
      chain.getHostCreds().getHost();
    } catch (MlflowClientException e) {
      Assert.assertTrue(e.getMessage().contains("Oh no!"), e.getMessage());
    }
  }
}
