package org.mlflow.client.creds;

/** A static hostname and optional credentials to talk to an MLflow server. */
public class BasicMlflowHostCreds implements MlflowHostCreds, MlflowHostCredsProvider {
  private String host;
  private String username;
  private String password;
  private String token;
  private boolean noTlsVerify;

  public BasicMlflowHostCreds(String host) {
    this.host = host;
  }

  public BasicMlflowHostCreds(String host, String username, String password) {
    this.host = host;
    this.username = username;
    this.password = password;
  }

  public BasicMlflowHostCreds(String host, String token) {
    this.host = host;
    this.token = token;
  }

  public BasicMlflowHostCreds(
      String host,
      String username,
      String password,
      String token,
      boolean noTlsVerify) {
    this.host = host;
    this.username = username;
    this.password = password;
    this.token = token;
    this.noTlsVerify = noTlsVerify;
  }

  @Override
  public String getHost() {
    return host;
  }

  @Override
  public String getUsername() {
    return username;
  }

  @Override
  public String getPassword() {
    return password;
  }

  @Override
  public String getToken() {
    return token;
  }

  @Override
  public boolean getNoTlsVerify() {
    return noTlsVerify;
  }

  @Override
  public MlflowHostCreds getHostCreds() {
    return this;
  }

  @Override
  public void refresh() {
    // no-op
  }
}
