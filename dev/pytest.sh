#!/bin/bash

fetch_labels() {
    # If this script is not running under GitHub Actions, exit
    if [ -z $GITHUB_ACTIONS ]; then
        echo ""
        return
    fi

    # If the event name is not "pull_request", exit
    if [ "$GITHUB_EVENT_NAME" != "pull_request" ]; then
        echo ""
        return
    fi

    # Read the PR data
    PR_DATA=$(cat $GITHUB_EVENT_PATH)

    # Get the PR number from the data
    PR_NUMBER=$(echo $PR_DATA | jq --raw-output .pull_request.number)

    # Use the GitHub API to fetch the labels
    LABELS=$(curl -s https://api.github.com/repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/labels | jq --raw-output .[].name)

    # Output the list of labels
    echo $LABELS
}

main() {
    LABELS=$(fetch_labels)

    # Check if the "fail-fast" label is applied to the PR
    if [[ $LABELS == *"fail-fast"* ]]; then
        EXTRA_OPTIONS="--exitfirst"
    fi

    pytest $EXTRA_OPTIONS "${@:1}"
}

main "$@"
