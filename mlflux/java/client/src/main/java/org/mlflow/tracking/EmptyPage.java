package org.mlflow.tracking;

import java.util.Collections;
import java.util.Optional;

public class EmptyPage<E> implements Page<E> {

  /**
   * Creates an empty page
   */
  EmptyPage() {}

  /**
   * @return Zero
   */
  public int getPageSize() {
    return 0;
  }

  /**
   * @return False
   */
  public boolean hasNextPage() {
    return false;
  }

  /**
   * @return An empty Optional.
   */
  public Optional<String> getNextPageToken() {
    return Optional.empty();
  }

  /**
   * @return An {@link org.mlflow.tracking.EmptyPage}
   */
  public EmptyPage getNextPage() {
    return this;
  }

  /**
   * @return An empty iterable.
   */
  public Iterable getItems() {
    return Collections.EMPTY_LIST;
  }

}
