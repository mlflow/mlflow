import { Button, CheckIcon, DropdownMenu, GearIcon, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import type { RunsChartsGlobalLineChartConfig } from '../../experiment-page/models/ExperimentPageUIState';
import { isUndefined } from 'lodash';
import { RunsChartsLineChartXAxisType } from './RunsCharts.common';
import { useCallback } from 'react';
import { LineSmoothSlider } from '../../LineSmoothSlider';
import { FormattedMessage, useIntl } from 'react-intl';
import type { RunsChartsUIConfigurationSetter } from '../hooks/useRunsChartsUIConfiguration';

export const RunsChartsGlobalChartSettingsDropdown = ({
  globalLineChartConfig,
  metricKeyList,
  updateUIState,
}: {
  metricKeyList: string[];
  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;
  updateUIState: (stateSetter: RunsChartsUIConfigurationSetter) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { lineSmoothness, selectedXAxisMetricKey, xAxisKey } = globalLineChartConfig || {};

  const updateGlobalLineChartSettings = useCallback(
    (newSettings: Partial<RunsChartsGlobalLineChartConfig>) =>
      updateUIState((state) => ({
        ...state,
        globalLineChartConfig: {
          ...state.globalLineChartConfig,
          ...newSettings,
        },
      })),
    [updateUIState],
  );

  const isUsingGlobalMetricXaxis = xAxisKey === RunsChartsLineChartXAxisType.METRIC;

  const label = intl.formatMessage({
    defaultMessage: 'Configure charts',
    description: 'Experiment page > view controls > global settings for line chart view > dropdown button label',
  });

  return (
    <DropdownMenu.Root modal={false}>
      <Tooltip
        componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_44"
        content={label}
      >
        <DropdownMenu.Trigger asChild>
          <Button
            componentId="mlflow.charts.controls.global_chart_setup_dropdown"
            icon={<GearIcon />}
            aria-label={label}
            css={{ flexShrink: 0 }}
          />
        </DropdownMenu.Trigger>
      </Tooltip>
      <DropdownMenu.Content align="end" css={{ minWidth: 300 }}>
        <DropdownMenu.Group
          role="region"
          aria-label={intl.formatMessage({
            defaultMessage: 'X-axis',
            description:
              'Experiment page > view controls > global settings for line chart view > settings for x-axis section label',
          })}
        >
          <DropdownMenu.Label css={{ display: 'flex', gap: 8 }}>
            <FormattedMessage
              defaultMessage="X-axis"
              description="Experiment page > view controls > global settings for line chart view > settings for x-axis section label"
            />
          </DropdownMenu.Label>
          <DropdownMenu.CheckboxItem
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_68"
            checked={xAxisKey === RunsChartsLineChartXAxisType.STEP}
            onClick={() => updateGlobalLineChartSettings({ xAxisKey: RunsChartsLineChartXAxisType.STEP })}
          >
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="Step"
              description="Experiment page > view controls > global settings for line chart view > settings for x-axis > label for setting to use step axis in all charts"
            />
          </DropdownMenu.CheckboxItem>
          <DropdownMenu.CheckboxItem
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_78"
            checked={xAxisKey === RunsChartsLineChartXAxisType.TIME}
            onClick={() => updateGlobalLineChartSettings({ xAxisKey: RunsChartsLineChartXAxisType.TIME })}
          >
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="Time (wall)"
              description="Experiment page > view controls > global settings for line chart view > settings for x-axis > label for setting to use wall time axis in all charts"
            />
          </DropdownMenu.CheckboxItem>
          <DropdownMenu.CheckboxItem
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_88"
            checked={xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE}
            onClick={() => updateGlobalLineChartSettings({ xAxisKey: RunsChartsLineChartXAxisType.TIME_RELATIVE })}
          >
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="Time (relative)"
              description="Experiment page > view controls > global settings for line chart view > settings for x-axis > label for setting to use relative time axis in all charts"
            />
          </DropdownMenu.CheckboxItem>
          <DropdownMenu.Sub>
            <DropdownMenu.SubTrigger
              css={{
                paddingLeft: theme.spacing.xs + theme.spacing.sm,
              }}
            >
              <DropdownMenu.IconWrapper>
                <CheckIcon
                  css={{
                    visibility: isUsingGlobalMetricXaxis ? 'visible' : 'hidden',
                  }}
                />
              </DropdownMenu.IconWrapper>
              <FormattedMessage
                defaultMessage="Metric"
                description="Experiment page > view controls > global settings for line chart view > settings for x-axis > label for setting to use metric axis in all charts"
              />
            </DropdownMenu.SubTrigger>
            <DropdownMenu.SubContent css={{ maxHeight: 300, overflow: 'auto' }}>
              {metricKeyList.map((metricKey) => (
                <DropdownMenu.CheckboxItem
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_runschartsglobalchartsettingsdropdown.tsx_118"
                  key={metricKey}
                  checked={selectedXAxisMetricKey === metricKey && isUsingGlobalMetricXaxis}
                  onClick={() =>
                    updateGlobalLineChartSettings({
                      xAxisKey: RunsChartsLineChartXAxisType.METRIC,
                      selectedXAxisMetricKey: metricKey,
                    })
                  }
                >
                  <DropdownMenu.ItemIndicator />
                  {metricKey}
                </DropdownMenu.CheckboxItem>
              ))}
            </DropdownMenu.SubContent>
          </DropdownMenu.Sub>
        </DropdownMenu.Group>
        <DropdownMenu.Group
          role="region"
          aria-label={intl.formatMessage({
            defaultMessage: 'Line smoothing',
            description:
              'Runs charts > line chart > configuration > label for line smoothing slider control. The control allows changing data trace line smoothness from 1 to 100, where 1 is the original data trace and 100 is the smoothest trace. Line smoothing helps eliminate noise in the data.',
          })}
        >
          <DropdownMenu.Label>
            <FormattedMessage
              defaultMessage="Line smoothing"
              description="Runs charts > line chart > configuration > label for line smoothing slider control. The control allows changing data trace line smoothness from 1 to 100, where 1 is the original data trace and 100 is the smoothest trace. Line smoothing helps eliminate noise in the data."
            />
          </DropdownMenu.Label>

          <div css={{ padding: theme.spacing.sm }}>
            <LineSmoothSlider
              min={0}
              max={100}
              onChange={(lineSmoothness) => updateGlobalLineChartSettings({ lineSmoothness })}
              value={lineSmoothness ? lineSmoothness : 0}
            />
          </div>
        </DropdownMenu.Group>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
