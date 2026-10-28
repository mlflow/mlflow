import {
  Button,
  GavelIcon,
  PencilIcon,
  Tag,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

export const AssessmentPaneToggle = ({
  assessmentCount = 0,
  compact = false,
}: {
  assessmentCount?: number;
  compact?: boolean;
}): JSX.Element | null => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { assessmentsPaneExpanded, setAssessmentsPaneExpanded, assessmentsPaneEnabled } =
    useModelTraceExplorerViewState();
  const hasAssessments = assessmentCount > 0;
  const label = hasAssessments
    ? intl.formatMessage(
        {
          defaultMessage: 'Assess trace ({assessmentCount, plural, one {# assessment} other {# assessments}})',
          description: 'Accessible label and tooltip for assessing a trace when the trace has assessments',
        },
        { assessmentCount },
      )
    : intl.formatMessage({
        defaultMessage: 'Assess trace',
        description: 'Accessible label and tooltip for assessing a trace',
      });

  if (assessmentsPaneExpanded) {
    return null;
  }

  return (
    <Tooltip componentId="shared.model-trace-explorer.assessments-pane-toggle.tooltip" content={label}>
      <span>
        <Button
          disabled={!assessmentsPaneEnabled}
          componentId="shared.model-trace-explorer.assessments-pane-toggle"
          aria-label={label}
          icon={<PencilIcon />}
          onClick={() => setAssessmentsPaneExpanded?.(true)}
          css={{
            '&:not(:hover):not(:focus-visible)': {
              borderColor: `${theme.colors.borderDecorative} !important`,
            },
          }}
        >
          {!compact && (
            <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Assess" description="Button text for assessing a trace" />
              {hasAssessments && (
                <Tag
                  color="indigo"
                  componentId="shared.model-trace-explorer.assessments-pane-toggle-count"
                  css={{ margin: 0 }}
                >
                  <GavelIcon />
                  <Typography.Text css={{ marginLeft: theme.spacing.xs }}>{assessmentCount}</Typography.Text>
                </Tag>
              )}
            </span>
          )}
          {compact && hasAssessments && assessmentCount}
        </Button>
      </span>
    </Tooltip>
  );
};
