# regal ignore:directory-package-mismatch
package mlflow_test

import rego.v1

test_redundant_default_pull_request_types_ignores_order if {
	test_input := {"true": {"pull_request": {"types": ["reopened", "opened", "synchronize"]}}}
	data.mlflow.deny_redundant_default_pull_request_types with input as test_input
}

test_redundant_default_pull_request_types_covers_both_triggers if {
	test_input := {"true": {
		"pull_request": {"types": ["opened", "synchronize", "reopened"]},
		"pull_request_target": {"types": ["reopened", "synchronize", "opened"]},
	}}
	messages := data.mlflow.deny_redundant_default_pull_request_types with input as test_input
	count(messages) == 2
}

test_redundant_default_pull_request_types_allows_omitted_types if {
	test_input := {"true": {"pull_request": {}}}
	messages := data.mlflow.deny_redundant_default_pull_request_types with input as test_input
	count(messages) == 0
}

test_redundant_default_pull_request_types_allows_subsets if {
	test_input := {"true": {"pull_request": {"types": ["opened", "reopened"]}}}
	messages := data.mlflow.deny_redundant_default_pull_request_types with input as test_input
	count(messages) == 0
}

test_redundant_default_pull_request_types_allows_additional_types if {
	test_input := {"true": {"pull_request_target": {"types": ["opened", "synchronize", "reopened", "closed"]}}}
	messages := data.mlflow.deny_redundant_default_pull_request_types with input as test_input
	count(messages) == 0
}
