import { isNil } from 'lodash';

import {
  ModelsIcon,
  ParagraphSkeleton,
  Typography,
  Tooltip,
  Tag,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useQuery } from '../../query-client/queryClient';

import { ErrorCell } from './ErrorCell';
import { NullCell } from './NullCell';
import { StackedComponents } from './StackedComponents';
import type { ModelTraceInfoV3 } from '../../model-trace-explorer/ModelTrace.types';
import { getAjaxUrl, makeRequest } from '../utils/FetchUtils';
import MlflowUtils from '../utils/MlflowUtils';
import { Link } from '../utils/RoutingUtils';
import { getExperimentIdFromTraceLocation } from '../utils/TraceUtils';

export const LoggedModelCell = (props: {
  currentTraceInfo?: ModelTraceInfoV3;
  otherTraceInfo?: ModelTraceInfoV3;
  isComparing: boolean;
}) => {
  const { currentTraceInfo, otherTraceInfo, isComparing } = props;
  const currentModelId = currentTraceInfo?.trace_metadata?.['mlflow.modelId'];
  const otherModelId = otherTraceInfo?.trace_metadata?.['mlflow.modelId'];

  return (
    <StackedComponents
      first={
        currentModelId ? (
          <LoggedModelComponent
            experimentId={getExperimentIdFromTraceLocation(currentTraceInfo?.trace_location)}
            modelId={currentModelId}
            isComparing={isComparing}
          />
        ) : (
          <NullCell isComparing={isComparing} />
        )
      }
      second={
        isComparing &&
        (otherModelId ? (
          <LoggedModelComponent
            experimentId={getExperimentIdFromTraceLocation(otherTraceInfo?.trace_location)}
            modelId={otherModelId}
            isComparing={isComparing}
          />
        ) : (
          <NullCell isComparing={isComparing} />
        ))
      }
    />
  );
};

const LoggedModelComponent = (props: { experimentId?: string; modelId: string; isComparing: boolean }) => {
  const { experimentId, modelId, isComparing } = props;
  const { theme } = useDesignSystemTheme();

  const { data, isLoading, error } = useLoggedModelName({ loggedModelId: modelId });
  const modelName = data?.info?.name;

  if (isLoading) {
    return <ParagraphSkeleton />;
  }

  if (error) {
    return <ErrorCell />;
  }

  if (!modelName) {
    return <NullCell isComparing={isComparing} />;
  }

  const content = (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        maxWidth: '100%',
      }}
    >
      <ModelsIcon css={{ color: theme.colors.textPrimary, fontSize: 16 }} />
      <Typography.Text css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {modelName}
      </Typography.Text>
    </div>
  );

  return (
    <Tooltip componentId="mlflow.eval-runs.model-version-cell-tooltip" content={modelName}>
      <Tag
        componentId="mlflow.eval-runs.model-version-cell"
        id="model-version-cell"
        css={{ width: 'fit-content', maxWidth: '100%', marginRight: 0, cursor: experimentId ? 'pointer' : undefined }}
      >
        {experimentId ? (
          <Link
            to={MlflowUtils.getLoggedModelPageRoute(experimentId, modelId)}
            target="_blank"
            css={{
              maxWidth: '100%',
              display: 'block',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
            title={modelName}
          >
            {content}
          </Link>
        ) : (
          <span
            css={{
              maxWidth: '100%',
              display: 'block',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
            title={modelName}
          >
            {content}
          </span>
        )}
      </Tag>
    </Tooltip>
  );
};

interface LoggedModelNameResponse {
  model: {
    info: {
      name?: string;
    };
  };
}

/**
 * Retrieve logged model from API based on its ID
 */
const useLoggedModelName = ({ loggedModelId }: { loggedModelId?: string }) => {
  const { data, isLoading, isFetching, refetch, error } = useQuery<LoggedModelNameResponse, Error>({
    queryKey: ['loggedModelName', loggedModelId],
    queryFn: async () => {
      const res: LoggedModelNameResponse = await makeRequest(
        getAjaxUrl(`ajax-api/2.0/mlflow/logged-models/${loggedModelId}`),
        'GET',
      );
      return res;
    },
    cacheTime: Infinity,
    staleTime: Infinity,
    refetchOnMount: false,
    retry: 1,
    enabled: !isNil(loggedModelId),
  });

  return {
    isLoading,
    isFetching,
    data: data?.model,
    refetch,
    error,
  } as const;
};
