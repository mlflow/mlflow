import { useIntl } from 'react-intl';

import { EvalRunsDatePreset } from './useEvalRunsFilters';

/** Display order for the Date presets, ALL last so "no filter" reads as the reset option. */
export const DATE_PRESET_ORDER: EvalRunsDatePreset[] = [
  EvalRunsDatePreset.LAST_HOUR,
  EvalRunsDatePreset.LAST_24_HOURS,
  EvalRunsDatePreset.LAST_7_DAYS,
  EvalRunsDatePreset.LAST_30_DAYS,
  EvalRunsDatePreset.LAST_YEAR,
  EvalRunsDatePreset.ALL,
];

/**
 * Human labels for the Date presets, used by the standalone Date button. Built with inline
 * `formatMessage` (rather than a `defineMessages` map indexed by enum) to keep FormatJS extraction
 * happy with the enum keys.
 */
export const useEvalRunsFilterLabels = () => {
  const intl = useIntl();

  const datePresetLabels: Record<EvalRunsDatePreset, string> = {
    [EvalRunsDatePreset.LAST_HOUR]: intl.formatMessage({
      defaultMessage: 'Last hour',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.LAST_24_HOURS]: intl.formatMessage({
      defaultMessage: 'Last 24 hours',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.LAST_7_DAYS]: intl.formatMessage({
      defaultMessage: 'Last 7 days',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.LAST_30_DAYS]: intl.formatMessage({
      defaultMessage: 'Last 30 days',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.LAST_YEAR]: intl.formatMessage({
      defaultMessage: 'Last year',
      description: 'Date range preset in the evaluation runs filters',
    }),
    [EvalRunsDatePreset.ALL]: intl.formatMessage({
      defaultMessage: 'All time',
      description: 'Date range preset in the evaluation runs filters',
    }),
  };

  return { datePresetLabels };
};
