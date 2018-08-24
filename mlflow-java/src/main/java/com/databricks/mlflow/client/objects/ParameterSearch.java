package com.databricks.mlflow.client.objects;

/** Convenience class for easier API search. */
public class ParameterSearch  extends BaseSearch {
    private String value;

    public ParameterSearch(String key, String comparator, String value) {
        super(key, comparator);
        this.value = value;
    }
    public String getValue() { return value; }

    @Override
    public String toString() {
        return
             "[key="+getKey()
             + " comparator="+getComparator()
             + " value="+value
             + "]"
        ;
    }
}
