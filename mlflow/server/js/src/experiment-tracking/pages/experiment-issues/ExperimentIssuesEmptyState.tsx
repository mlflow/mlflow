import { type CSSObject } from '@emotion/react';
import {
  BracketsErrorIcon,
  Button,
  DollarIcon,
  Empty,
  RobotIcon,
  SparkleIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { Link } from '../../../common/utils/RoutingUtils';
import { IssueDetectionIllustration } from './IssueDetectionIllustration';
import { IssueDetectionSettingsControl } from './IssueDetectionSettingsControl';
import { JobStatus } from '../../components/run-page/hooks/useFetchJobStatus';

const IssueDetectionExampleBubbles = () => {
  const { theme } = useDesignSystemTheme();
  const bubbleStyles: CSSObject = {
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
    width: 'fit-content',
    maxWidth: '100%',
    minWidth: 0,
    boxSizing: 'border-box',
    padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
    color: theme.colors.textSecondary,
    backgroundColor: theme.colors.backgroundPrimary,
    border: `1px solid ${theme.colors.borderDecorative}`,
    borderRadius: theme.borders.borderRadiusFull,
  };
  const highlightStyles = {
    borderRadius: theme.borders.borderRadiusSm,
    padding: `0 ${theme.spacing.xs / 2}px`,
  };
  const warningHighlightStyles = {
    ...highlightStyles,
    backgroundColor: theme.isDarkMode ? theme.colors.yellow800 : theme.colors.yellow200,
  };
  const errorHighlightStyles = {
    ...highlightStyles,
    backgroundColor: theme.isDarkMode ? theme.colors.red800 : theme.colors.red200,
  };

  return (
    <div
      aria-hidden="true"
      css={{
        width: theme.responsive.breakpoints.sm,
        maxWidth: '100%',
        marginBottom: theme.spacing.md,
        containerType: 'inline-size',
      }}
    >
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr)',
          gridTemplateAreas: '"cost" "illustration" "agent" "tool"',
          alignItems: 'center',
          gap: theme.spacing.sm,
          [`@container (min-width: ${theme.responsive.breakpoints.sm}px)`]: {
            gridTemplateColumns: 'minmax(0, 1fr) auto minmax(0, 1fr)',
            gridTemplateAreas: '"cost cost ." ". illustration agent" "tool tool ."',
          },
        }}
      >
        <div css={{ gridArea: 'illustration', justifySelf: 'center' }}>
          <IssueDetectionIllustration />
        </div>
        <ul css={{ display: 'contents', margin: 0, padding: 0, listStyle: 'none', textAlign: 'start' }}>
          <li
            css={{
              ...bubbleStyles,
              gridArea: 'cost',
              justifySelf: 'start',
              opacity: 0.7,
              [`@container (min-width: ${theme.responsive.breakpoints.sm}px)`]: {
                marginLeft: theme.spacing.md,
              },
            }}
          >
            <DollarIcon />
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Cost increased <highlight>3×</highlight> last week"
                description="Experiment issues tab > Empty state > Illustrative cost issue"
                values={{
                  highlight: (chunks) => <span css={warningHighlightStyles}>{chunks}</span>,
                }}
              />
            </Typography.Text>
          </li>
          <li
            css={{
              ...bubbleStyles,
              gridArea: 'agent',
              justifySelf: 'end',
              opacity: 0.6,
            }}
          >
            <RobotIcon />
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Agent response <highlight>violated policy</highlight>"
                description="Experiment issues tab > Empty state > Illustrative agent quality issue"
                values={{
                  highlight: (chunks) => <span css={warningHighlightStyles}>{chunks}</span>,
                }}
              />
            </Typography.Text>
          </li>
          <li
            css={{
              ...bubbleStyles,
              gridArea: 'tool',
              justifySelf: 'start',
              opacity: 0.5,
              [`@container (min-width: ${theme.responsive.breakpoints.sm}px)`]: {
                marginLeft: theme.spacing.xl,
              },
            }}
          >
            <BracketsErrorIcon />
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Tool error: <highlight>invalid argument</highlight>"
                description="Experiment issues tab > Empty state > Illustrative tool call issue"
                values={{
                  highlight: (chunks) => <span css={errorHighlightStyles}>{chunks}</span>,
                }}
              />
            </Typography.Text>
          </li>
        </ul>
      </div>
    </div>
  );
};

export interface ExperimentIssuesEmptyStateProps {
  experimentId: string;
  isInList?: boolean;
  detectionStatus?: JobStatus;
  isCleanScan?: boolean;
  isDetecting: boolean;
  detectionEnabled: boolean;
  onRunIssueDetection: () => void;
  /**
   * History link for a stalled submission without an active run. Accepted runs use toolbar history.
   */
  runHistoryRoute?: string;
}

export const ExperimentIssuesEmptyState = ({
  experimentId,
  isInList = false,
  detectionStatus,
  isCleanScan = false,
  isDetecting,
  detectionEnabled,
  onRunIssueDetection,
  runHistoryRoute,
}: ExperimentIssuesEmptyStateProps) => {
  const { theme } = useDesignSystemTheme();
  const secondaryLinkCss = {
    '&&, &&:focus, &&:hover, &&:active': {
      color: theme.colors.textSecondary,
      '.anticon': {
        color: theme.colors.textSecondary,
      },
    },
  } as const;

  return (
    <Empty
      dangerouslyAppendEmotionCSS={{ transform: `translateY(-${theme.spacing.md}px)` }}
      image={isInList ? undefined : <IssueDetectionExampleBubbles />}
      title={
        <span css={{ display: 'block', marginBottom: theme.spacing.xs }}>
          {isInList ? (
            <FormattedMessage
              defaultMessage="No issues recorded"
              description="Experiment issues tab > Empty list title after background issue detection"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Discover issues automatically"
              description="Experiment issues tab > Empty state title before detection has been run"
            />
          )}
        </span>
      }
      description={
        <>
          {!isInList ? (
            <FormattedMessage
              defaultMessage="Scan your traces to surface failure modes, quality regressions, and cost anomalies, then triage and fix them in one place."
              description="Experiment issues tab > Empty state description before detection has been run"
            />
          ) : detectionStatus === JobStatus.TIMEOUT ? (
            <FormattedMessage
              defaultMessage="The previous issue detection timed out. Start a new detection to try again."
              description="Experiment issues tab > Empty state description after issue detection timed out"
            />
          ) : detectionStatus === JobStatus.FAILED ? (
            <FormattedMessage
              defaultMessage="The previous issue detection failed. Start a new detection to try again."
              description="Experiment issues tab > Empty state description after issue detection failed"
            />
          ) : detectionStatus === JobStatus.CANCELED ? (
            <FormattedMessage
              defaultMessage="The previous issue detection was canceled. Start a new detection to try again."
              description="Experiment issues tab > Empty state description after issue detection was canceled"
            />
          ) : isCleanScan ? (
            <FormattedMessage
              defaultMessage="Issue detection did not find problems in the analyzed traces. Run it again as your application changes."
              description="Experiment issues tab > Empty state description after a completed detection found no issues"
            />
          ) : (
            <FormattedMessage
              defaultMessage="Issue detection has completed. There are currently no issues in this experiment. Run it again as your application changes."
              description="Experiment issues tab > Completed detection with an empty store and unknown or nonzero historical results"
            />
          )}
          {isInList && detectionEnabled && (
            <div>
              <FormattedMessage
                defaultMessage="To improve your next detection, try <priorities>setting detection priorities</priorities>."
                description="Experiment issues tab > Empty list hint recommending detection priorities"
                values={{
                  priorities: (chunks) => (
                    <IssueDetectionSettingsControl experimentId={experimentId} linkText={chunks} />
                  ),
                }}
              />
            </div>
          )}
        </>
      }
      button={
        isInList ? undefined : (
          <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: theme.spacing.sm }}>
            {detectionEnabled && (
              <Button
                componentId="mlflow.experiment-issues.empty-state.run-detection"
                type="primary"
                icon={<SparkleIcon />}
                loading={isDetecting}
                disabled={isDetecting}
                aria-busy={isDetecting}
                onClick={onRunIssueDetection}
              >
                <FormattedMessage
                  defaultMessage="Run issue detection"
                  description="Experiment issues tab > Empty state action to run issue detection"
                />
              </Button>
            )}
            {/* On one line so the stack stays two rows tall, and in the same muted grey, so neither
              link competes with the action above them. */}
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md }}>
              {runHistoryRoute && (
                <Link
                  componentId="mlflow.experiment-issues.empty-state.run-history"
                  to={runHistoryRoute}
                  css={secondaryLinkCss}
                >
                  <FormattedMessage
                    defaultMessage="View run history"
                    description="Experiment issues tab > Empty state link to the detection run history"
                  />
                </Link>
              )}
              <Typography.Link
                componentId="mlflow.experiment-issues.empty-state.learn-how"
                href="https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues/"
                openInNewTab
                css={secondaryLinkCss}
              >
                <FormattedMessage
                  defaultMessage="Learn how it works"
                  description="Experiment issues tab > Empty state documentation link"
                />
              </Typography.Link>
            </div>
          </div>
        )
      }
    />
  );
};
