import {
  Checkbox,
  Input,
  Switch,
  useDesignSystemTheme,
  LegacyTooltip,
  InfoSmallIcon,
  LegacyInfoTooltip,
} from '@databricks/design-system';
import type { RunsChartsCardConfig, RunsChartsDifferenceCardConfig } from '../../runs-charts.types';
import { DISABLED_GROUP_WHEN_GROUPBY, DifferenceCardConfigCompareGroup } from '../../runs-charts.types';
import { RunsChartsConfigureField } from './RunsChartsConfigure.common';
import { useCallback } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';

/**
 * Form containing configuration controls for runs compare difference view chart.
 */
export const RunsChartsConfigureDifferenceChart = ({
  state,
  onStateChange,
  metricKeyList,
  paramKeyList,
  groupBy,
}: {
  metricKeyList: string[];
  paramKeyList: string[];
  state: Partial<RunsChartsDifferenceCardConfig>;
  onStateChange: (setter: (current: RunsChartsCardConfig) => RunsChartsDifferenceCardConfig) => void;
  groupBy: RunsGroupByConfig | null;
}) => {
  /**
   * Callback for updating compare groups
   */
  const updateCompareGroups = useCallback(
    (compareGroup: DifferenceCardConfigCompareGroup) => {
      onStateChange((current) => {
        const currentConfig = current as RunsChartsDifferenceCardConfig;
        const compareGroups = currentConfig.compareGroups;
        if (compareGroups.includes(compareGroup)) {
          return {
            ...(current as RunsChartsDifferenceCardConfig),
            compareGroups: compareGroups.filter((group) => group !== compareGroup),
          };
        } else {
          return { ...(current as RunsChartsDifferenceCardConfig), compareGroups: [...compareGroups, compareGroup] };
        }
      });
    },
    [onStateChange],
  );

  const updateChartName = useCallback(
    (e) => onStateChange((current) => ({ ...(current as RunsChartsDifferenceCardConfig), chartName: e.target.value })),
    [onStateChange],
  );

  const updateShowChangeFromBaseline = useCallback(
    (showChangeFromBaseline: boolean) =>
      onStateChange((current) => ({
        ...(current as RunsChartsDifferenceCardConfig),
        showChangeFromBaseline,
      })),
    [onStateChange],
  );

  const updateShowDifferencesOnly = useCallback(
    (showDifferencesOnly: boolean) =>
      onStateChange((current) => ({
        ...(current as RunsChartsDifferenceCardConfig),
        showDifferencesOnly,
      })),
    [onStateChange],
  );

  const { theme } = useDesignSystemTheme();
  const { formatMessage } = useIntl();

  return (
    <>
      <RunsChartsConfigureField
        title={formatMessage({
          defaultMessage: 'Compare',
          description:
            'Runs charts > components > config > RunsChartsConfigureDifferenceChart > Compare config section',
        })}
      >
        <Checkbox.Group id="checkbox-group" defaultValue={state.compareGroups}>
          {Object.values(DifferenceCardConfigCompareGroup).map((group) => {
            const groupedCondition = groupBy ? DISABLED_GROUP_WHEN_GROUPBY.includes(group) : false;
            return (
              <div css={{ display: 'inline-flex', alignItems: 'center' }} key={group}>
                <Checkbox
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfiguredifferencechart.tsx_98"
                  key={group}
                  value={group}
                  isChecked={state.compareGroups?.includes(group)}
                  onChange={() => updateCompareGroups(group)}
                  disabled={groupedCondition}
                >
                  {group}
                </Checkbox>
                {groupedCondition && (
                  <LegacyInfoTooltip
                    title={
                      <FormattedMessage
                        defaultMessage="Disable grouped runs to compare"
                        description="Experiment tracking > components > runs-charts > RunsChartsConfigureDifferenceCharts > disable grouped runs info message"
                      />
                    }
                  />
                )}
              </div>
            );
          })}
        </Checkbox.Group>
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            padding: `${theme.spacing.md}px 0px`,
            gap: theme.spacing.sm,
          }}
        >
          <Switch
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfiguredifferencechart.tsx_129"
            checked={state.showChangeFromBaseline}
            onChange={updateShowChangeFromBaseline}
            label={formatMessage({
              defaultMessage: 'Show change from baseline',
              description:
                'Runs charts > components > config > RunsChartsConfigureDifferenceChart > Show change from baseline toggle',
            })}
          />
          <Switch
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfiguredifferencechart.tsx_138"
            checked={state.showDifferencesOnly}
            onChange={updateShowDifferencesOnly}
            label={formatMessage({
              defaultMessage: 'Show differences only',
              description:
                'Runs charts > components > config > RunsChartsConfigureDifferenceChart > Show differences only toggle',
            })}
          />
        </div>
      </RunsChartsConfigureField>

      <RunsChartsConfigureField
        title={formatMessage({
          defaultMessage: 'Chart name',
          description:
            'Runs charts > components > config > RunsChartsConfigureDifferenceChart > Chart name config section',
        })}
      >
        <Input
          componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_config_runschartsconfiguredifferencechart.tsx_157"
          value={state.chartName}
          onChange={updateChartName}
        />
      </RunsChartsConfigureField>
    </>
  );
};
