import http from "k6/http";

export default function () {
  // Set base url from environment variable, the variable ' -e HOSTNAME = xxxx ' must be added to command line argument
  const base_url = "http://" + __ENV.HOSTNAME + "/api/2.0/mlflow";

  const experiment_response = http.post(
    `${base_url}/experiments/create`,
    JSON.stringify({
      name: `exp_k6_${Date.now()}`,
      tags: [
        {
          key: "description",
          value: "k6 experiment",
        },
      ],
    }),
    {
      headers: {
        "Content-Type": "application/json",
      },
    }
  );

  const experiment_id = experiment_response.json().experiment_id;

  const run_response = http.post(
    `${base_url}/runs/create`,
    JSON.stringify({
      experiment_id: experiment_id,
      start_time: Date.now(),
      tags: [
        {
          key: "mlflow.user",
          value: "k6",
        },
      ],
    }),
    {
      headers: {
        "Content-Type": "application/json",
      },
    }
  );

  // retrieve run id
  const runId = run_response.json().run.info.run_id;

  // 100 tags
  const tags = Array.from(Array(100), (_, i) => {
    return {
      key: `tag_${i}`,
      value: `tag_value_${i}`,
    };
  });

  // 100 params
  const params = Array.from(Array(100), (_, i) => {
    return {
      key: `param_${i}`,
      value: `param_value_${i}`,
    };
  });

  // 800 metrics
  const metrics = Array.from(Array(200), (_, i) => {
    return {
      key: `metric_${i}`,
      value: i * Math.random(),
      timestamp: Date.now(),
      step: 0,
    };
  });

  // log batch
  http.post(
    `${base_url}/runs/log-batch`,
    JSON.stringify({
      run_id: runId,
      tags: tags,
      params: params,
      metrics: metrics,
    }),
    {
      headers: {
        "Content-Type": "application/json",
      },
    }
  );
}
