import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { useLocation, useNavigate } from '../../../common/utils/RoutingUtils';
import { ExperimentPageTabName } from '../../constants';
import Routes from '../../routes';
import { CancelIssueDetectionButton } from './CancelIssueDetectionButton';
import { IssueDetectionIllustration } from './IssueDetectionIllustration';

export interface ExperimentIssuesDetectionProgressProps {
  experimentId: string;
  /** Ends the in-progress run; unset while there is none to end yet. */
  onCancelDetection?: () => Promise<void>;
}

/**
 * The Issues tab while a detection runs, in place of the empty list.
 *
 * Detection is a server job, so it outlives this page: the panel says so, and offers the two
 * things a waiting user actually wants — somewhere to go meanwhile, and a way out of a run that
 * has stopped reporting progress.
 */
export const ExperimentIssuesDetectionProgress = ({
  experimentId,
  onCancelDetection,
}: ExperimentIssuesDetectionProgressProps) => {
  const { theme } = useDesignSystemTheme();
  const { search } = useLocation();
  const navigate = useNavigate();
  const tracesRoute = {
    pathname: Routes.getExperimentPageTabRoute(experimentId, ExperimentPageTabName.Traces),
    search,
  };

  return (
    <section
      role="status"
      aria-live="polite"
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        maxWidth: theme.spacing.lg * 20,
        textAlign: 'center',
      }}
    >
      <div css={{ marginBottom: theme.spacing.md }}>
        <IssueDetectionIllustration />
      </div>
      <Typography.Title level={3} color="secondary" withoutMargins>
        <FormattedMessage
          defaultMessage="Detecting issues..."
          description="Experiment issues tab > In-progress issue detection state title"
        />
      </Typography.Title>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          marginTop: theme.spacing.xs,
        }}
      >
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Scanning your traces for issues. This can take a while."
            description="Experiment issues tab > In-progress issue detection state description"
          />
        </Typography.Text>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Detection runs on the server, so you can leave this page and return later."
            description="Experiment issues tab > In-progress issue detection guidance about leaving the page"
          />
        </Typography.Text>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="If it has stopped making progress, cancel to start a new detection."
            description="Experiment issues tab > In-progress issue detection guidance on when to cancel"
          />
        </Typography.Text>
      </div>
      <div css={{ display: 'flex', gap: theme.spacing.sm, marginTop: theme.spacing.md }}>
        <CancelIssueDetectionButton
          componentId="mlflow.experiment-issues.detection-progress.cancel"
          onCancel={onCancelDetection}
        />
        <Button
          componentId="mlflow.experiment-issues.detection-progress.go-to-traces"
          onClick={() => navigate(tracesRoute)}
        >
          <FormattedMessage
            defaultMessage="Go to Traces"
            description="Experiment issues tab > In-progress issue detection action to go to the traces tab"
          />
        </Button>
      </div>
    </section>
  );
};
