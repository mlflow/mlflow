package org.mlflow.tracking.creds;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.mlflow.tracking.MlflowClientException;

public class HostCredsProviderChain implements MlflowHostCredsProvider {
  private static final Logger logger = LoggerFactory.getLogger(HostCredsProviderChain.class);

  private final List<MlflowHostCredsProvider> hostCredsProviders = new ArrayList<>();

  public HostCredsProviderChain(MlflowHostCredsProvider... hostCredsProviders) {
    this.hostCredsProviders.addAll(Arrays.asList(hostCredsProviders));
  }

  @Override
  public MlflowHostCreds getHostCreds() {
    List<String> exceptionMessages = new ArrayList<>();
    for (MlflowHostCredsProvider provider : hostCredsProviders) {
      try {
        MlflowHostCreds hostCreds = provider.getHostCreds();

        if (hostCreds != null && hostCreds.getHost() != null) {
          logger.debug("Loading credentials from " + provider.toString());
          return hostCreds;
        }
      } catch (Exception e) {
        String message = provider + ": " + e.getMessage();
        logger.debug("Unable to load credentials from " + message);
        exceptionMessages.add(message);
      }
    }
    throw new MlflowClientException("Unable to load MLflow Host/Credentials from any provider in" +
      " the chain: " + exceptionMessages);
  }

  @Override
  public void refresh() {
    for (MlflowHostCredsProvider provider : hostCredsProviders) {
      provider.refresh();
    }
  }
}