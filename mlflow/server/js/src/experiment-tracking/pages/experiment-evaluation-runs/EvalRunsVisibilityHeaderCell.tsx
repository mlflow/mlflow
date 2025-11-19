import { DashIcon, DropdownMenu, VisibleIcon, VisibleOffIcon, useDesignSystemTheme } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import React from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { RUNS_VISIBILITY_MODE } from '../../components/experiment-page/models/ExperimentPageUIState';
import { useExperimentEvaluationRunsRowVisibility } from './hooks/useExperimentEvaluationRunsRowVisibility';

/**
 * Header cell component for the visibility column in evaluation runs table.
 * Displays an eye icon button that opens a dropdown menu with visibility mode options.
 */
export const EvalRunsVisibilityHeaderCell = React.memo(() => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { visibilityMode, setVisibilityMode, usingCustomVisibility, allRunsHidden } =
    useExperimentEvaluationRunsRowVisibility();

  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <button
          css={styles.actionButton(theme)}
          data-testid="eval-runs-visibility-column-header"
          aria-label={intl.formatMessage({
            defaultMessage: 'Toggle visibility of evaluation runs',
            description: 'Evaluation runs table > toggle visibility of runs > accessible label',
          })}
        >
          {visibilityMode === RUNS_VISIBILITY_MODE.HIDEALL ||
          visibilityMode === RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS ||
          allRunsHidden ? (
            <VisibleOffIcon />
          ) : (
            <VisibleIcon />
          )}
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Content>
        <DropdownMenu.RadioGroup
          componentId="mlflow.eval-runs.visibility-mode-selector"
          value={visibilityMode}
          onValueChange={(value) => setVisibilityMode(value as RUNS_VISIBILITY_MODE)}
        >
          <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.FIRST_10_RUNS}>
            {/* Dropdown menu does not support indeterminate state, so we're doing it manually */}
            <DropdownMenu.ItemIndicator>{usingCustomVisibility ? <DashIcon /> : null}</DropdownMenu.ItemIndicator>
            <FormattedMessage
              defaultMessage="Show first 10"
              description="Menu option for showing only 10 first runs in the evaluation runs table"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.FIRST_20_RUNS}>
            <DropdownMenu.ItemIndicator>{usingCustomVisibility ? <DashIcon /> : null}</DropdownMenu.ItemIndicator>
            <FormattedMessage
              defaultMessage="Show first 20"
              description="Menu option for showing only 20 first runs in the evaluation runs table"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.SHOWALL}>
            <DropdownMenu.ItemIndicator>{usingCustomVisibility ? <DashIcon /> : null}</DropdownMenu.ItemIndicator>
            <FormattedMessage
              defaultMessage="Show all runs"
              description="Menu option for revealing all hidden runs in the evaluation runs table"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.HIDEALL}>
            <DropdownMenu.ItemIndicator>{usingCustomVisibility ? <DashIcon /> : null}</DropdownMenu.ItemIndicator>
            <FormattedMessage
              defaultMessage="Hide all runs"
              description="Menu option for hiding all runs in the evaluation runs table"
            />
          </DropdownMenu.RadioItem>
          <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS}>
            <DropdownMenu.ItemIndicator>{usingCustomVisibility ? <DashIcon /> : null}</DropdownMenu.ItemIndicator>
            <FormattedMessage
              defaultMessage="Hide finished runs"
              description="Menu option for hiding all finished runs in the evaluation runs table"
            />
          </DropdownMenu.RadioItem>
        </DropdownMenu.RadioGroup>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
});

const styles = {
  actionButton: (theme: Theme) => ({
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    padding: 0,
    margin: 0,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    verticalAlign: 'text-bottom',
    minWidth: 24,
    minHeight: 24,
    svg: {
      width: theme.general.iconFontSize,
      height: theme.general.iconFontSize,
      cursor: 'pointer',
      color: theme.colors.textPrimary,
      flexShrink: 0,
    },
  }),
};
