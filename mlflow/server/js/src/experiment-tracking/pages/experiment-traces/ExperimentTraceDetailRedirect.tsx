import { useEffect } from 'react';
import { useNavigate, useParams, useSearchParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG } from '../../constants';

/**
 * Redirects clean trace detail URLs to the query-param-based URL that opens the trace drawer.
 *
 * /experiments/:experimentId/traces/:traceId?o=...
 *   → /experiments/:experimentId/traces?selectedEvaluationId=<fullId>&o=...
 *
 * For V3 traces (tr-xxx), the selectedEvaluationId is the traceId as-is.
 * For V4 traces (hex ID without tr- prefix), we look up the experiment's UC schema tag
 * to build the full identifier: trace:/catalog.schema/traceId.
 */
const ExperimentTraceDetailRedirect = () => {
  const { experimentId, traceId } = useParams<'experimentId' | 'traceId'>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const isV3 = traceId?.startsWith('tr-');

  const { data: experimentData, loading } = useGetExperimentQuery({
    experimentId,
    options: { skip: isV3 },
  });

  useEffect(() => {
    if (!traceId || !experimentId) return;

    // For V3, redirect immediately
    // For V4, wait for experiment data to resolve the UC schema
    if (!isV3 && loading) return;

    let selectedEvaluationId = traceId;
    if (!isV3) {
      const tags = experimentData && 'tags' in experimentData ? experimentData?.tags : [];
      const ucSchema = tags?.find((tag) => tag.key === MLFLOW_EXPERIMENT_TRACE_STORAGE_UC_SCHEMA_TAG)?.value;
      if (ucSchema) {
        selectedEvaluationId = `trace:/${ucSchema}/${traceId}`;
      }
    }

    const params = new URLSearchParams(searchParams);
    params.set('selectedEvaluationId', selectedEvaluationId);
    const target = `${Routes.getExperimentPageTracesTabRoute(experimentId)}?${params.toString()}`;
    navigate(target, { replace: true });
  }, [experimentId, traceId, isV3, loading, experimentData, searchParams, navigate]);

  return null;
};

export default ExperimentTraceDetailRedirect;
