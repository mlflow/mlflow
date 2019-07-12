package org.mlflow.tracking;

import java.util.List;
import java.util.Optional;

public interface Page<E> {

    /**
     * @return Returns the number of elements in this page.
     */
    public int getPageSize();

    /**
     * @return Returns true if there are more pages that can be retrieved from the API.
     */
    public boolean hasNextPage();

    /**
     * @return Returns an Optional of the token string to get the next page. 
     * Empty if there is no next page.
     */
    public Optional<String> getNextPageToken();

    /**
     * @return Retrieves the next Page object using the next page token,
     * or returns an empty page if there are no more pages.
     */
    public Page<E> getNextPage();

    /**
     * @return Returns a List of the elements in this Page.
     */
    public List<E> getItems();

}