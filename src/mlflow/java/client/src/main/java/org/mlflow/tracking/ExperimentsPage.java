package org.mlflow.tracking;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Optional;
import org.mlflow.api.proto.Service.*;

public class ExperimentsPage implements Page<Experiment> {

  private final String token;
  private final List<Experiment> experiments;

  private final MlflowClient client;
  private final String searchFilter;
  private final ViewType experimentViewType;
  private final List<String> orderBy;
  private final int maxResults;

  /**
   * Creates a fixed size page of Experiments.
   */
  ExperimentsPage(List<Experiment> experiments,
                  String token,
                  String searchFilter,
                  ViewType experimentViewType,
                  int maxResults,
                  List<String> orderBy,
                  MlflowClient client) {
    this.experiments = Collections.unmodifiableList(experiments);
    this.token = token;
    this.searchFilter = searchFilter;
    this.experimentViewType = experimentViewType;
    this.orderBy = orderBy;
    this.maxResults = maxResults;
    this.client = client;
  }

  /**
   * @return The number of experiments in the page.
   */
  public int getPageSize() {
    return this.experiments.size();
  }

  /**
   * @return True if a token for the next page exists and isn't empty. Otherwise returns false.
   */
  public boolean hasNextPage() {
    return this.token != null && this.token != "";
  }

  /**
   * @return An optional with the token for the next page. 
   * Empty if the token doesn't exist or is empty.
   */
  public Optional<String> getNextPageToken() {
    if (this.hasNextPage()) {
      return Optional.of(this.token);
    } else {
      return Optional.empty();
    }
  }

  /**
   * @return The next page of experiments matching the search criteria.
   * If there are no more pages, an {@link org.mlflow.tracking.EmptyPage} will be returned.
   */
  public Page<Experiment> getNextPage() {
    if (this.hasNextPage()) {
      return this.client.searchExperiments(this.searchFilter,
                                           this.experimentViewType,
                                           this.maxResults,
                                           this.orderBy,
                                           this.token);
    } else {
      return new EmptyPage();
    }
  }

  /**
   * @return An iterable over the experiments in this page.
   */
  public List<Experiment> getItems() {
    return experiments;
  }

}
