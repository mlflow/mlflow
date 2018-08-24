package com.databricks.mlflow.client.objects;

/** Convenience class for easier API search. */
public class MetricSearch extends BaseSearch {
    private Float value;

    public MetricSearch(String key, String comparator, Float value) {
        super(key, comparator);
        this.value = value;
    }
    public Float getValue() { return value; }

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
