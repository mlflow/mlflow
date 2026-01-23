import { TracesV3Toolbar } from '../../components/experiment-page/components/traces-v3/TracesV3Toolbar';
import invariant from 'invariant';
import { useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { useCallback, useMemo, useState } from 'react';
import type { RowSelectionState } from '@tanstack/react-table';
import { useRegisterSelectedIds } from '@mlflow/mlflow/src/assistant';
import {
  SESSION_COLUMN_ID,
  TracesTableColumnType,
  INPUTS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
} from '@databricks/web-shared/genai-traces-table';
import type { TracesTableColumn } from '@databricks/web-shared/genai-traces-table';
import { MonitoringConfigProvider, useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { useMonitoringFiltersTimeRange } from '../../hooks/useMonitoringFilters';
import { getChatSessionsFilter } from './utils';
import { ExperimentChatSessionsPageWrapper } from './ExperimentChatSessionsPageWrapper';
import { TracesV3Logs } from '../../components/experiment-page/components/traces-v3/TracesV3Logs';
import { useDesignSystemTheme } from '@databricks/design-system';

const defaultCustomDefaultSelectedColumns = (column: TracesTableColumn) => {
  if (column.type === TracesTableColumnType.ASSESSMENT || column.type === TracesTableColumnType.EXPECTATION) {
    return true;
  }
  return [SESSION_COLUMN_ID, INPUTS_COLUMN_ID, RESPONSE_COLUMN_ID].includes(column.id);
};

const ExperimentChatSessionsPageImpl = () => {
  const { experimentId } = useParams();
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  useRegisterSelectedIds('selectedSessionIds', rowSelection);
  invariant(experimentId, 'Experiment ID must be defined');

  const { loading: isLoadingExperiment } = useGetExperimentQuery({
    experimentId,
  });

  const timeRange = useMonitoringFiltersTimeRange();

  const traceSearchLocations = useMemo(
    () => {
      return [createTraceLocationForExperiment(experimentId)];
    },
    // prettier-ignore
    [
      experimentId,
    ],
  );

  const filters = useMemo(() => getChatSessionsFilter({ sessionId: null }), []);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, gap: theme.spacing.sm }}>
      <TracesV3Toolbar
        // prettier-ignore
        viewState="sessions"
      />
      <TracesV3Logs
        experimentId={experimentId}
        additionalFilters={filters}
        endpointName=""
        timeRange={timeRange}
        customDefaultSelectedColumns={defaultCustomDefaultSelectedColumns}
        forceGroupBySession
      />
    </div>
  );
};

const ExperimentChatSessionsPage = () => {
  return (
    <ExperimentChatSessionsPageWrapper>
      <MonitoringConfigProvider>
        <ExperimentChatSessionsPageImpl />
      </MonitoringConfigProvider>
    </ExperimentChatSessionsPageWrapper>
  );
};

export default ExperimentChatSessionsPage;
