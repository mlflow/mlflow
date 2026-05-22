import { DOC_BASE } from '@mlflow/mlflow/src/common/constants-databricks';
import DatabricksUtils from '@mlflow/mlflow/src/common/utils/DatabricksUtils';
import { CloudProvider } from '@mlflow/mlflow/src/shared/constants-databricks';

/**
 * Build a docs URL under the `mlflow3/genai/eval-monitor/` namespace for the current cloud.
 * Falls back to AWS when the cloud provider is unknown so we never render a broken link.
 */
export const getEvalMonitorDocsLink = (suffix: string): string => {
  const cloud = DatabricksUtils.getCloudProvider() ?? CloudProvider.AWS;
  const base = DOC_BASE[cloud] ?? DOC_BASE[CloudProvider.AWS];
  return `${base}/mlflow3/genai/eval-monitor/${suffix}`;
};
