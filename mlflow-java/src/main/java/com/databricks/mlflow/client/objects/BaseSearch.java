package com.databricks.mlflow.client.objects;

/** Convenience class for easier API search. */
public class BaseSearch {
    private String key;
    private String comparator;

    public BaseSearch(String key, String comparator) {
        this.key = key;
        this.comparator = comparator;
    }
    public String getKey() { return key; }
    public String getComparator() { return comparator; }

    @Override
    public String toString() {
        return
             "[key="+key
             + " comparator="+comparator
             + "]"
        ;
    }
}
