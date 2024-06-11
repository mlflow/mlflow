import http from "k6/http";

// Set base url from environment variable, the variable ' -e MLFLOW_TRACKING_URI = xxxx ' must be added to command line argument
const base_url = __ENV.MLFLOW_TRACKING_URI + "/api/2.0/mlflow";

if (!base_url.startsWith("http")) {
  throw new Error("MLFLOW_TRACKING_URI must be a valid URL, starting with http(s)");
}

export function setup() {
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

  return run_response.json().run.info.run_id;
}

function logBatch({ runId, tagCount, paramCount, metricCount }) {
  const tags = Array.from(Array(tagCount), (_, i) => {
    return {
      key: `tag_${i}`,
      value: `tag_value_${i}`,
    };
  });

  const params = Array.from(Array(paramCount), (_, i) => {
    return {
      key: `param_${i}`,
      value: `param_value_${i}`,
    };
  });

  const metrics = Array.from(Array(metricCount), (_, i) => {
    return {
      key: `metric_${i}`,
      value: i * Math.random(),
      timestamp: Date.now(),
      step: 0,
    };
  });

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

export default function (runId) {
  logBatch({ runId, tagCount: 5, paramCount: 3, metricCount: 10 });
  // logBatch({ runId, tagCount: 50, paramCount: 50, metricCount: 300 });
  // logBatch({ runId, tagCount: 100, paramCount: 100, metricCount: 800 });
  // logBatch({ runId, tagCount: 0, paramCount: 10, metricCount: 0 });
}
