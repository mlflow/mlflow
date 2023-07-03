# MLflow AI Gateway

The examples provided within this directory show how to get started with individual providers and at least
one of the supported route types. When configuring an instance of the MLflow AI Gateway, multiple providers,
instances of route types, and model versions can be specified for each query route on the Gateway.

To get started, see the individual examples for the providers that you are interested in creating interfaces with.

**Note**: The examples and README files contained within this directory have been generated through
the use of the MLflow AI Gateway to interface with LLM providers, using minimal prompting and subsequent editing.

## Example configuration files

Within this directory are example config files for each of the supported providers. If using these as a guide
for configuring a large number of routes, ensure that the placeholder names (i.e., "completions", "chat", "embeddings")
are modified to prevent collisions. These names are provided for clarity only for the examples and real-world
use cases should define a relevant and meaningful route name to eliminate ambiguity and minimize the chances of name collisions.
