package org.mlflow.tracking.creds;

import java.util.AbstractMap;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.testng.Assert;
import org.testng.annotations.Test;

public class DatabricksDynamicHostCredsProviderTest {
  private static Map<String, String> baseMap = new HashMap<>();
  public static class MyDynamicProvider extends AbstractMap<String, String> {
    @Override
    public Set<Map.Entry<String, String>> entrySet() {
      return baseMap.entrySet();
    }
  }

  @Test
  public void testUpdatesAfterPut() {
    baseMap.put("host", "hello");
    MlflowHostCredsProvider provider = DatabricksDynamicHostCredsProvider.createIfAvailable(
      MyDynamicProvider.class.getName());
    Assert.assertNotNull(provider);
    Assert.assertEquals(provider.getHostCreds().getHost(), "hello");
    Assert.assertNull(provider.getHostCreds().getToken());

    baseMap.put("token", "toke");
    Assert.assertEquals(provider.getHostCreds().getHost(), "hello");
    Assert.assertEquals(provider.getHostCreds().getToken(), "toke");

    baseMap.put("token", "toke2");
    Assert.assertEquals(provider.getHostCreds().getHost(), "hello");
    Assert.assertEquals(provider.getHostCreds().getToken(), "toke2");
  }

  @Test
  public void testUsernamePassword() {
    baseMap.put("host", "hello");
    baseMap.put("username", "boop");
    baseMap.put("password", "beep");
    MlflowHostCredsProvider provider = DatabricksDynamicHostCredsProvider.createIfAvailable(
      MyDynamicProvider.class.getName());
    Assert.assertNotNull(provider);
    Assert.assertEquals(provider.getHostCreds().getHost(), "hello");
    Assert.assertEquals(provider.getHostCreds().getUsername(), "boop");
    Assert.assertEquals(provider.getHostCreds().getPassword(), "beep");
  }

  @Test
  public void testTlsInsecure() {
    baseMap.put("host", "hello");
    MlflowHostCredsProvider provider = DatabricksDynamicHostCredsProvider.createIfAvailable(
      MyDynamicProvider.class.getName());
    Assert.assertNotNull(provider);
    Assert.assertEquals(provider.getHostCreds().getHost(), "hello");
    Assert.assertFalse(provider.getHostCreds().shouldIgnoreTlsVerification());

    baseMap.put("shouldIgnoreTlsVerification", "true");
    Assert.assertTrue(provider.getHostCreds().shouldIgnoreTlsVerification());

    baseMap.put("shouldIgnoreTlsVerification", "false");
    Assert.assertFalse(provider.getHostCreds().shouldIgnoreTlsVerification());
  }
}

