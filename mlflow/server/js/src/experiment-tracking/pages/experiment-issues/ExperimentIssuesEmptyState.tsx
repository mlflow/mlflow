import { keyframes, type CSSObject } from '@emotion/react';
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
import { FormattedMessage, useIntl } from 'react-intl';

const revealBlock = keyframes({
  '0%': { opacity: 0, transform: 'translateY(25%)' },
  '7%': { opacity: 1, transform: 'translateY(0)' },
  '100%': { opacity: 1, transform: 'translateY(0)' },
});

const revealDetectionMarker = keyframes({
  '0%': { opacity: 0, transform: 'translateY(-50%) scale(0.8)' },
  '4%': { opacity: 1, transform: 'translateY(8%) scale(1.05)' },
  '7%': { opacity: 1, transform: 'translateY(0) scale(1)' },
  '100%': { opacity: 1, transform: 'translateY(0) scale(1)' },
});

const ANIMATION_DURATION_MULTIPLIER = 1.3;
const ANIMATION_CYCLE_DURATION_MULTIPLIER = 32;

interface IsometricBlock {
  x: number;
  y: number;
  isAnomaly?: boolean;
}

const ISOMETRIC_BLOCKS: IsometricBlock[] = [
  { x: 36, y: 34 },
  { x: 27, y: 39 },
  { x: 45, y: 39 },
  { x: 18, y: 44 },
  { x: 36, y: 44 },
  { x: 54, y: 44 },
  { x: 27, y: 49 },
  { x: 45, y: 49 },
  { x: 36, y: 54 },
  { x: 36, y: 19 },
  { x: 27, y: 24 },
  { x: 45, y: 24 },
  { x: 36, y: 29, isAnomaly: true },
];

const getBlockFaces = ({ x, y }: IsometricBlock) => ({
  top: `${x},${y} ${x + 9},${y + 5} ${x},${y + 10} ${x - 9},${y + 5}`,
  left: `${x - 9},${y + 5} ${x},${y + 10} ${x},${y + 17} ${x - 9},${y + 12}`,
  right: `${x + 9},${y + 5} ${x},${y + 10} ${x},${y + 17} ${x + 9},${y + 12}`,
});

const IssueDetectionIllustration = () => {
  const { theme } = useDesignSystemTheme();
  const revealDuration = theme.animation.transitionDuration * 2 * ANIMATION_DURATION_MULTIPLIER;
  const revealDelay = Math.round((theme.animation.transitionDuration / 10) * ANIMATION_DURATION_MULTIPLIER);
  const animationCycleDuration =
    theme.animation.transitionDuration * ANIMATION_CYCLE_DURATION_MULTIPLIER * ANIMATION_DURATION_MULTIPLIER;

  return (
    <div
      css={{
        width: theme.spacing.lg * 3,
        height: theme.spacing.lg * 3,
      }}
    >
      <svg viewBox="0 0 72 72" width="100%" height="100%" focusable="false">
        {ISOMETRIC_BLOCKS.map((block, index) => {
          const faces = getBlockFaces(block);
          const outlineColor = block.isAnomaly ? theme.colors.textValidationDanger : theme.colors.actionDisabledText;
          const topColor = block.isAnomaly ? theme.colors.backgroundDanger : theme.colors.backgroundPrimary;
          const leftColor = block.isAnomaly ? theme.colors.backgroundDanger : theme.colors.backgroundSecondary;
          const rightColor = block.isAnomaly ? theme.colors.borderDanger : theme.colors.actionDisabledBackground;

          return (
            <g
              key={`${block.x}-${block.y}`}
              css={{
                transformBox: 'fill-box',
                transformOrigin: 'center',
                '@media (prefers-reduced-motion: no-preference)': {
                  animation: `${revealBlock} ${animationCycleDuration}ms ease-out infinite both`,
                  animationDelay: `${index * revealDelay}ms`,
                },
              }}
              stroke={outlineColor}
              strokeWidth="1.25"
              strokeLinejoin="round"
            >
              <polygon points={faces.top} fill={topColor} />
              <polygon points={faces.left} fill={leftColor} />
              <polygon points={faces.right} fill={rightColor} />
            </g>
          );
        })}
        <g
          css={{
            transformBox: 'fill-box',
            transformOrigin: 'center',
            '@media (prefers-reduced-motion: no-preference)': {
              animation: `${revealDetectionMarker} ${animationCycleDuration}ms ease-out infinite both`,
              animationDelay: `${revealDuration}ms`,
            },
          }}
          fill={theme.colors.backgroundPrimary}
          stroke={theme.colors.actionPrimaryBackgroundDefault}
          strokeWidth="1.5"
          strokeLinecap="round"
        >
          <circle cx="56" cy="16" r="6" />
          <path d="M60.5 20.5 65 25" />
        </g>
      </svg>
    </div>
  );
};

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
  hasCompletedDetectionRuns: boolean;
  isCleanScan?: boolean;
  hasTimedOutDetectionRun: boolean;
  hasFailedDetectionRun: boolean;
  isDetecting: boolean;
  detectionEnabled: boolean;
  onRunIssueDetection: () => void;
}

export const ExperimentIssuesEmptyState = ({
  hasCompletedDetectionRuns,
  isCleanScan = false,
  hasTimedOutDetectionRun,
  hasFailedDetectionRun,
  isDetecting,
  detectionEnabled,
  onRunIssueDetection,
}: ExperimentIssuesEmptyStateProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const buttonText =
    hasTimedOutDetectionRun || hasFailedDetectionRun
      ? intl.formatMessage({
          defaultMessage: 'Start new detection',
          description: 'Experiment issues tab > Empty state action after issue detection failed or timed out',
        })
      : intl.formatMessage({
          defaultMessage: 'Run issue detection',
          description: 'Experiment issues tab > Empty state action to run issue detection',
        });

  return (
    <Empty
      dangerouslyAppendEmotionCSS={{ transform: `translateY(-${theme.spacing.md}px)` }}
      image={<IssueDetectionExampleBubbles />}
      title={
        <span css={{ display: 'block', marginBottom: theme.spacing.xs }}>
          {isCleanScan ? (
            <FormattedMessage
              defaultMessage="No issues detected"
              description="Experiment issues tab > Empty state title after a completed detection found no issues"
            />
          ) : hasCompletedDetectionRuns ? (
            <FormattedMessage
              defaultMessage="No issues to display"
              description="Experiment issues tab > Empty state title after prior detection"
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
        hasTimedOutDetectionRun ? (
          <FormattedMessage
            defaultMessage="The previous issue detection timed out. Start a new detection to try again."
            description="Experiment issues tab > Empty state description after issue detection timed out"
          />
        ) : hasFailedDetectionRun ? (
          <FormattedMessage
            defaultMessage="The previous issue detection failed. Start a new detection to try again."
            description="Experiment issues tab > Empty state description after issue detection failed"
          />
        ) : isCleanScan ? (
          <FormattedMessage
            defaultMessage="Issue detection did not find problems in the analyzed traces. Run it again as your application changes."
            description="Experiment issues tab > Empty state description after a completed detection found no issues"
          />
        ) : hasCompletedDetectionRuns ? (
          <FormattedMessage
            defaultMessage="Issue detection has completed. There are currently no issues in this experiment. Run it again as your application changes."
            description="Experiment issues tab > Completed detection with an empty store and unknown or nonzero historical results"
          />
        ) : (
          <FormattedMessage
            defaultMessage="Scan your traces to surface failure modes, quality regressions, and cost anomalies, then triage and fix them in one place."
            description="Experiment issues tab > Empty state description before detection has been run"
          />
        )
      }
      button={
        <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: theme.spacing.sm }}>
          {detectionEnabled && (
            <Button
              componentId="mlflow.experiment-issues.empty-state.run-detection"
              type="primary"
              icon={<SparkleIcon />}
              loading={isDetecting}
              disabled={isDetecting}
              aria-busy={isDetecting}
              aria-label={buttonText}
              onClick={onRunIssueDetection}
            >
              {buttonText}
            </Button>
          )}
          <Typography.Link
            componentId="mlflow.experiment-issues.empty-state.learn-how"
            href="https://mlflow.org/docs/latest/genai/eval-monitor/ai-insights/detect-issues/"
            openInNewTab
            css={{
              '&&, &&:focus, &&:hover, &&:active': {
                color: theme.colors.textSecondary,
                '.anticon': {
                  color: theme.colors.textSecondary,
                },
              },
            }}
          >
            <FormattedMessage
              defaultMessage="Learn how it works"
              description="Experiment issues tab > Empty state documentation link"
            />
          </Typography.Link>
        </div>
      }
    />
  );
};
