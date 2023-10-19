#!/bin/bash

# pytest runner to fail fast if the "fail-fast" label is present on the PR.

fetch_labels() {
    if [ -z $GITHUB_ACTIONS ]; then
        echo ""
        return
    fi

    if [ "$GITHUB_EVENT_NAME" != "pull_request" ]; then
        echo ""
        return
    fi

    PR_DATA=$(cat $GITHUB_EVENT_PATH)
    PR_NUMBER=$(echo $PR_DATA | jq --raw-output .pull_request.number)
    LABELS=$(curl -s https://api.github.com/repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/labels | jq --raw-output .[].name)
    echo $LABELS
}

main() {
    LABELS=$(fetch_labels)

    if [[ $LABELS == *"fail-fast"* ]]; then
        EXTRA_OPTIONS="--exitfirst"
    fi

    echo "pytest $EXTRA_OPTIONS ${@:1}"
    pytest $EXTRA_OPTIONS "${@:1}"
}

main "$@"
