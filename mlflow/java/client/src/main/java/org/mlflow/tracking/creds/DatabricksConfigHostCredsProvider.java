package org.mlflow.tracking.creds;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

import org.ini4j.Ini;
import org.ini4j.Profile;

public class DatabricksConfigHostCredsProvider extends DatabricksHostCredsProvider {
  private static final String CONFIG_FILE_ENV_VAR = "DATABRICKS_CONFIG_FILE";

  private final String profile;

  private DatabricksMlflowHostCreds hostCreds;

  public DatabricksConfigHostCredsProvider(String profile) {
    this.profile = profile;
  }

  public DatabricksConfigHostCredsProvider() {
    this.profile = null;
  }

  private void loadConfigIfNecessary() {
    if (hostCreds == null) {
      reloadConfig();
    }
  }

  private void reloadConfig() {
    String basePath = System.getenv(CONFIG_FILE_ENV_VAR);
    if (basePath == null) {
      String userHome = System.getProperty("user.home");
      basePath = Paths.get(userHome, ".databrickscfg").toString();
    }

    if (!new File(basePath).isFile()) {
      throw new IllegalStateException("Could not find Databricks configuration file" +
        " (" + basePath + "). Please run 'databricks configure' using the Databricks CLI.");
    }

    Ini ini;
    try {
      ini = new Ini(new File(basePath));
    } catch (IOException e) {
      throw new IllegalStateException("Failed to load databrickscfg file at " + basePath, e);
    }

    Profile.Section section;
    if (profile == null) {
      section = ini.get("DEFAULT");
      if (section == null) {
        throw new IllegalStateException("Could not find 'DEFAULT' section within config file" +
          " (" + basePath + "). Please run 'databricks configure' using the Databricks CLI.");
      }
    } else {
      section = ini.get(profile);
      if (section == null) {
        throw new IllegalStateException("Could not find '" + profile + "' section within config" +
          " file  (" + basePath + "). Please run 'databricks configure --profile " + profile + "'" +
          " using the Databricks CLI.");
      }
    }
    assert (section != null);

    String host = section.get("host");
    String username = section.get("username");
    String password = section.get("password");
    String token = section.get("token");
    boolean insecure = section.get("insecure", "false").toLowerCase().equals("true");

    if (host == null) {
      throw new IllegalStateException("No 'host' configured within Databricks config file" +
        " (" + basePath + "). Please run 'databricks configure' using the Databricks CLI.");
    }

    boolean hasValidUserPassword = username != null && password != null;
    boolean hasValidToken = token != null;
    if (!hasValidUserPassword && !hasValidToken) {
      throw new IllegalStateException("No authentication configured within Databricks config file" +
        " (" + basePath + "). Please run 'databricks configure' using the Databricks CLI.");
    }

    this.hostCreds = new DatabricksMlflowHostCreds(host, username, password, token, insecure);
  }

  @Override
  public DatabricksMlflowHostCreds getHostCreds() {
    loadConfigIfNecessary();
    return hostCreds;
  }

  @Override
  public void refresh() {
    reloadConfig();
  }
}
