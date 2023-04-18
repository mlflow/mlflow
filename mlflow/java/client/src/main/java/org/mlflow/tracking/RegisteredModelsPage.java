package org.mlflow.tracking;

import org.mlflow.api.proto.ModelRegistry.RegisteredModel;

import java.util.Collections;
import java.util.List;
import java.util.Optional;

class RegisteredModelsPage implements Page<RegisteredModel> {

  private final String token;
  private final List<RegisteredModel> items;

  private final MlflowClient client;
  private final String searchFilter;
  private final List<String> orderBy;
  private final int maxResults;

  /**
   * Creates a fixed size page of RegisteredModels.
   */
  RegisteredModelsPage(List<RegisteredModel> models,
                       String token,
                       String searchFilter,
                       int maxResults,
                       List<String> orderBy,
                       MlflowClient client) {
    this.items = Collections.unmodifiableList(models);
    this.token = token;
    this.searchFilter = searchFilter;
    this.orderBy = orderBy;
    this.maxResults = maxResults;
    this.client = client;
  }

  /**
   * @return The number of registered models in the page.
   */
  public int getPageSize() {
    return this.items.size();
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
   * @return The next page of registered models matching the search criteria.
   * If there are no more pages, an {@link EmptyPage} will be returned.
   */
  public Page<RegisteredModel> getNextPage() {
    if (this.hasNextPage()) {
      return this.client.searchRegisteredModels(this.searchFilter,
                                                this.maxResults,
                                                this.orderBy,
                                                this.token);
    } else {
      return new EmptyPage<>();
    }
  }

  /**
   * @return An iterable over the registered models in this page.
   */
  public List<RegisteredModel> getItems() {
    return items;
  }

}
