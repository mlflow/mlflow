import {
  Button,
  CalendarIcon,
  ChevronDownIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxCustomButtonTriggerWrapper,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  XCircleFillIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useIntl } from 'react-intl';

import type { EvalRunsDatePreset } from './hooks/useEvalRunsFilters';
import { EVAL_RUNS_DEFAULT_DATE_PRESET_V2 } from './hooks/useEvalRunsFilters';
import { DATE_PRESET_ORDER, useEvalRunsFilterLabels } from './hooks/useEvalRunsFilterLabels';

export interface EvalRunsDateFilterButtonProps {
  datePreset: EvalRunsDatePreset;
  setDatePreset: (preset: EvalRunsDatePreset) => void;
}

/**
 * Standalone date-range control. Lives outside the Filters funnel — date range is the most-used
 * filter and earns an always-visible slot; it also avoids nesting two dropdowns in one popover.
 */
export const EvalRunsDateFilterButton = ({ datePreset, setDatePreset }: EvalRunsDateFilterButtonProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { datePresetLabels } = useEvalRunsFilterLabels();

  // Keep the trigger text to the selected preset while exposing a concise accessible name.
  const dateLabel = intl.formatMessage({
    defaultMessage: 'Date',
    description: 'Accessible label for the evaluation runs date filter button',
  });
  const clearSelectionLabel = intl.formatMessage({
    defaultMessage: 'Clear selection',
    description: 'Accessible label for clearing the evaluation runs date filter',
  });

  // Show clear only when non-default; clearing resets to the default.
  const isNonDefault = datePreset !== EVAL_RUNS_DEFAULT_DATE_PRESET_V2;

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, flexShrink: 0 }}>
      <DialogCombobox
        componentId="mlflow.eval-runs.filter-date-preset"
        label={dateLabel}
        value={[datePresetLabels[datePreset]]}
      >
        <DialogComboboxCustomButtonTriggerWrapper>
          <Button
            componentId="mlflow.eval-runs.filter-date-preset.trigger"
            icon={<CalendarIcon css={{ color: theme.colors.textSecondary }} />}
            endIcon={<ChevronDownIcon />}
            aria-label={dateLabel}
            css={{ flexShrink: 0 }}
          >
            {datePresetLabels[datePreset]}
          </Button>
        </DialogComboboxCustomButtonTriggerWrapper>
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            {DATE_PRESET_ORDER.map((preset) => (
              <DialogComboboxOptionListSelectItem
                key={preset}
                value={datePresetLabels[preset]}
                checked={datePreset === preset}
                onChange={() => setDatePreset(preset)}
              >
                {datePresetLabels[preset]}
              </DialogComboboxOptionListSelectItem>
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>
      {isNonDefault && (
        <Button
          componentId="mlflow.eval-runs.filter-date-preset.clear"
          type="tertiary"
          icon={<XCircleFillIcon />}
          aria-label={clearSelectionLabel}
          onClick={() => setDatePreset(EVAL_RUNS_DEFAULT_DATE_PRESET_V2)}
        />
      )}
    </div>
  );
};
