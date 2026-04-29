package org.mlflow.tracking.creds;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;

import org.apache.commons.configuration2.INIConfiguration;
import org.apache.commons.configuration2.SubnodeConfiguration;
import org.apache.commons.configuration2.ex.ConfigurationException;

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

    INIConfiguration ini;
    try {
      ini = new INIConfiguration();
      try (FileReader reader = new FileReader(basePath)) {
        ini.read(reader);
      }
    } catch (IOException | ConfigurationException e) {
      throw new IllegalStateException("Failed to load databrickscfg file at " + basePath, e);
    }

    SubnodeConfiguration section;
    if (profile == null) {
      section = ini.getSection("DEFAULT");
      if (section == null || section.isEmpty()) {
        throw new IllegalStateException("Could not find 'DEFAULT' section within config file" +
          " (" + basePath + "). Please run 'databricks configure' using the Databricks CLI.");
      }
    } else {
      section = ini.getSection(profile);
      if (section == null || section.isEmpty()) {
        throw new IllegalStateException("Could not find '" + profile + "' section within config" +
          " file  (" + basePath + "). Please run 'databricks configure --profile " + profile + "'" +
          " using the Databricks CLI.");
      }
    }
    assert (section != null);

    String host = section.getString("host");
    String username = section.getString("username");
    String password = section.getString("password");
    String token = section.getString("token");
    boolean insecure = "true".equalsIgnoreCase(section.getString("insecure", "false"));

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
