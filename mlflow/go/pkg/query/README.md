# Search Query Syntax

Mlflow has a [query syntax](https://mlflow.org/docs/latest/search-runs.html#search-query-syntax-deep-dive).

This package is meant to lex and parse this query dialect.

The code is slightly based on the https://github.com/tlaceby/parser-series.
I did not implement a proper Pratt parser because of how limited the query language is.
