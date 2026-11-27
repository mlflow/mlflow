import { Tag, Tooltip } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { ExperimentPageTabName } from '../../../../constants';
import type { ExperimentEntity } from '../../../../types';
import { useGetExperimentPageActiveTabByRoute } from '../../hooks/useGetExperimentPageActiveTabByRoute';
import { formatTraceArchivalRetentionForDisplay } from '../../../../../common/utils/traceArchival';

/**
 * "Archive after: <retention>" badge, shown on the traces and sessions tabs when the experiment has a trace
 * archival retention.
 */
export const ExperimentViewTraceArchivalBadge = ({ experiment }: { experiment: ExperimentEntity }) => {
  const intl = useIntl();
  const { tabName: activeTabByRoute } = useGetExperimentPageActiveTabByRoute();
  const traceArchivalRetention =
    formatTraceArchivalRetentionForDisplay(experiment.effectiveTraceArchivalRetention?.trim(), intl) || undefined;
  const shouldShowTraceArchivalBadge =
    activeTabByRoute === ExperimentPageTabName.Traces ||
    activeTabByRoute === ExperimentPageTabName.ChatSessions ||
    activeTabByRoute === ExperimentPageTabName.SingleChatSession;

  if (!traceArchivalRetention || !shouldShowTraceArchivalBadge) {
    return null;
  }

  return (
    <div css={{ display: 'flex', alignItems: 'center' }}>
      <Tooltip
        componentId="mlflow.experiment_view.header.trace_archival_badge_tooltip"
        content={
          <FormattedMessage
            defaultMessage="Archived traces remain accessible, but searching within span content no longer works after archival."
            description="Tooltip explaining the trace archival badge behavior"
          />
        }
      >
        <Tag componentId="mlflow.experiment_view.header.trace_archival_badge" color="turquoise" css={{ margin: 0 }}>
          <FormattedMessage
            defaultMessage="Archive after: {retention}"
            description="Badge showing the active trace archival retention for the experiment"
            values={{ retention: traceArchivalRetention }}
          />
        </Tag>
      </Tooltip>
    </div>
  );
};
