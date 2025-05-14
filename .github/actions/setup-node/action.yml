name: "setup-node"
description: "Set up Node"
inputs:
  node-version:
    description: "Node version to use."
    default: 20
    required: false

runs:
  using: "composite"
  steps:
    - uses: actions/setup-node@cdca7365b2dadb8aad0a33bc7601856ffabcc48e # v4.3.0
      with:
        node-version: ${{ inputs.node-version }}
