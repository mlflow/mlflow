import { useCallback, useEffect } from 'react';
import { useIntl } from 'react-intl';
import {
  extractCanonicalSortKey,
  isCanonicalSortKeyOfType,
  makeCanonicalSortKey,
} from '../../experiment-page/utils/experimentPage.column-utils';
import type { RunsCompareCardConfig, RunsCompareScatterCardConfig } from '../runs-compare.types';
import {
  RunsCompareMetricParamSelect,
  RunsCompareConfigureField,
  RunsCompareRunNumberSelect,
  runsCompareRunCountDefaultOptions,
} from './RunsCompareConfigure.common';

type ValidAxis = keyof Pick<RunsCompareScatterCardConfig, 'xaxis' | 'yaxis'>;

/**
 * Form containing configuration controls for scatter runs compare chart.
 */
export const RunsCompareConfigureScatterChart = ({
  state,
  onStateChange,
  metricKeyList,
  paramKeyList,
}: {
  metricKeyList: string[];
  paramKeyList: string[];
  state: RunsCompareScatterCardConfig;
  onStateChange: (setter: (current: RunsCompareCardConfig) => RunsCompareScatterCardConfig) => void;
}) => {
  const { formatMessage } = useIntl();

  /**
   * Callback for updating X or Y axis
   */
  const updateAxis = useCallback(
    (canonicalKey: string, axis: ValidAxis) => {
      const type = isCanonicalSortKeyOfType(canonicalKey, 'METRIC') ? 'METRIC' : 'PARAM';
      const key = extractCanonicalSortKey(canonicalKey, type);
      onStateChange((current) => ({
        ...(current as RunsCompareScatterCardConfig),
        [axis]: { key, type },
      }));
    },
    [onStateChange],
  );

  /**
   * Callback for updating run count
   */
  const updateVisibleRunCount = useCallback(
    (runsCountToCompare: number) => {
      onStateChange((current) => ({
        ...(current as RunsCompareScatterCardConfig),
        runsCountToCompare,
      }));
    },
    [onStateChange],
  );

  /**
   * If somehow axes are not predetermined, automatically
   * select the first metric/param so it's not empty
   */
  useEffect(() => {
    const firstMetric = metricKeyList?.[0];
    const firstParam = paramKeyList?.[0];
    if (!state.xaxis?.key) {
      if (firstMetric) {
        updateAxis(makeCanonicalSortKey('METRIC', firstMetric), 'xaxis');
      } else if (firstParam) {
        updateAxis(makeCanonicalSortKey('PARAM', firstParam), 'xaxis');
      }
    }
    if (!state.yaxis?.key) {
      if (firstMetric) {
        updateAxis(makeCanonicalSortKey('METRIC', firstMetric), 'yaxis');
      } else if (firstParam) {
        updateAxis(makeCanonicalSortKey('PARAM', firstParam), 'yaxis');
      }
    }
  }, [state.xaxis, state.yaxis, updateAxis, metricKeyList, paramKeyList]);

  return (
    <>
      <RunsCompareConfigureField
        title={formatMessage({
          defaultMessage: 'X axis',
          description:
            'Label for X axis in scatter chart configurator in compare runs chart config modal',
        })}
      >
        <RunsCompareMetricParamSelect
          value={state.xaxis.key ? makeCanonicalSortKey(state.xaxis.type, state.xaxis.key) : ''}
          onChange={(value) => {
            updateAxis(value, 'xaxis');
          }}
          paramKeyList={paramKeyList}
          metricKeyList={metricKeyList}
        />
      </RunsCompareConfigureField>
      <RunsCompareConfigureField
        title={formatMessage({
          defaultMessage: 'Y axis',
          description:
            'Label for Y axis in scatter chart configurator in compare runs chart config modal',
        })}
      >
        <RunsCompareMetricParamSelect
          value={state.yaxis.key ? makeCanonicalSortKey(state.yaxis.type, state.yaxis.key) : ''}
          onChange={(value) => {
            updateAxis(value, 'yaxis');
          }}
          paramKeyList={paramKeyList}
          metricKeyList={metricKeyList}
        />
      </RunsCompareConfigureField>
      <RunsCompareRunNumberSelect
        value={state.runsCountToCompare}
        onChange={updateVisibleRunCount}
        options={runsCompareRunCountDefaultOptions}
      />
    </>
  );
};
