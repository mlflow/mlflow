package org.mlflow.tracking;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import org.mlflow.api.proto.ModelRegistry.*;

public class ModelVersionsPage implements Page<ModelVersion> {

  private final String token;
  private final List<ModelVersion> mvs;

  private final MlflowClient client;
  private final String searchFilter;
  private final List<String> orderBy;
  private final int maxResults;

  /**
   * Creates a fixed size page of ModelVersions.
   */
  ModelVersionsPage(List<ModelVersion> mvs,
                    String token,
                    String searchFilter,
                    int maxResults,
                    List<String> orderBy,
                    MlflowClient client) {
    this.mvs = Collections.unmodifiableList(mvs);
    this.token = token;
    this.searchFilter = searchFilter;
    this.orderBy = orderBy;
    this.maxResults = maxResults;
    this.client = client;
  }

  /**
   * @return The number of model versions in the page.
   */
  public int getPageSize() {
    return this.mvs.size();
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
   * @return The next page of model versions matching the search criteria.
   * If there are no more pages, an {@link org.mlflow.tracking.EmptyPage} will be returned.
   */
  public Page<ModelVersion> getNextPage() {
    if (this.hasNextPage()) {
      return this.client.searchModelVersions(this.searchFilter,
                                             this.maxResults,
                                             this.orderBy,
                                             this.token);
    } else {
      return new EmptyPage();
    }
  }

  /**
   * @return An iterable over the model versions in this page.
   */
  public List<ModelVersion> getItems() {
    return mvs;
  }

}
