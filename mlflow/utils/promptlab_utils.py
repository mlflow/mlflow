import json


def create_eval_results_json(prompt_parameters, model_input, model_output_parameters, model_output):
    columns = [param.key for param in prompt_parameters] + ["prompt", "output"]
    data = [param.value for param in prompt_parameters] + [model_input, model_output]

    updated_columns = columns + [param.key for param in model_output_parameters]
    updated_data = data + [param.value for param in model_output_parameters]

    eval_results = {"columns": updated_columns, "data": [updated_data]}

    eval_results_json = json.dumps(eval_results)
    return eval_results_json
