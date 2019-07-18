package org.mlflow.tracking;

import java.lang.Iterable;
import java.util.Optional;

public interface Page<E> {

  /**
   * @return The number of elements in this page.
   */
  public int getPageSize();

  /**
   * @return True if there are more pages that can be retrieved from the API.
   */
  public boolean hasNextPage();

  /**
   * @return An Optional of the token string to get the next page. 
   * Empty if there is no next page.
   */
  public Optional<String> getNextPageToken();

  /**
   * @return Retrieves the next Page object using the next page token,
   * or returns an empty page if there are no more pages.
   */
  public Page<E> getNextPage();

  /**
   * @return A List of the elements in this Page.
   */
  public Iterable<E> getItems();

}
