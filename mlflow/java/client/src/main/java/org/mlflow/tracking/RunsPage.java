package org.mlflow.tracking;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Optional;
import org.mlflow.api.proto.Service.*;

public class RunsPage implements Page<Run> {

  private final String token;
  private final List<Run> runs;

  private final MlflowClient client;
  private final List<String> experimentIds;
  private final String searchFilter;
  private final ViewType runViewType;
  private final List<String> orderBy;
  private final int maxResults;

  /**
   * Creates a fixed size page of Runs.
   */
  RunsPage(List<Run> runs,
                  String token,
                  List<String> experimentIds,
                  String searchFilter,
                  ViewType runViewType,
                  int maxResults,
                  List<String> orderBy,
                  MlflowClient client) {
    this.runs = Collections.unmodifiableList(runs);
    this.token = token;
    this.experimentIds = experimentIds;
    this.searchFilter = searchFilter;
    this.runViewType = runViewType;
    this.orderBy = orderBy;
    this.maxResults = maxResults;
    this.client = client;
  }

  /**
   * @return The number of runs in the page.
   */
  public int getPageSize() {
    return this.runs.size();
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
   * @return The next page of runs matching the search criteria. 
   * If there are no more pages, an {@link org.mlflow.tracking.EmptyPage} will be returned.
   */
  public Page<Run> getNextPage() {
    if (this.hasNextPage()) {
      return this.client.searchRuns(this.experimentIds,
                                    this.searchFilter,
                                    this.runViewType,
                                    this.maxResults,
                                    this.orderBy,
                                    this.token);
    } else {
      return new EmptyPage();
    }
  }

  /**
   * @return An iterable over the runs in this page.
   */
  public List<Run> getItems() {
    return runs;
  }

}
