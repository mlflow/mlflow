import { useCallback, useState } from 'react';
import { useUpdateRunsChartsUIConfiguration } from '../hooks/useRunsChartsUIConfiguration';
import { useIntl } from 'react-intl';
import { Input, SearchIcon, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { useDebouncedCallback } from 'use-debounce';

export const RunsChartsFilterInput = ({ chartsSearchFilter }: { chartsSearchFilter?: string }) => {
  const updateChartsUIState = useUpdateRunsChartsUIConfiguration();
  const { theme } = useDesignSystemTheme();

  const [inputValue, setInputValue] = useState(() => chartsSearchFilter ?? '');
  const [searching, setSearching] = useState(false);

  const { formatMessage } = useIntl();

  const updateChartFilter = useCallback(
    (value: string) => {
      updateChartsUIState((current) => ({
        ...current,
        chartsSearchFilter: value,
      }));
      setSearching(false);
    },
    [updateChartsUIState],
  );

  const updateChartFilterDebounced = useDebouncedCallback(updateChartFilter, 150);

  return (
    <Input
      componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsfilterinput.tsx_30"
      role="searchbox"
      prefix={
        <div css={{ width: theme.general.iconFontSize, lineHeight: 0 }}>
          {searching ? <Spinner size="small" /> : <SearchIcon />}
        </div>
      }
      value={inputValue}
      allowClear
      onChange={(e) => {
        setInputValue(e.target.value);
        setSearching(true);
        updateChartFilterDebounced(e.target.value);
      }}
      placeholder={formatMessage({
        defaultMessage: 'Search metric charts',
        description: 'Run page > Charts tab > Filter metric charts input > placeholder',
      })}
    />
  );
};
