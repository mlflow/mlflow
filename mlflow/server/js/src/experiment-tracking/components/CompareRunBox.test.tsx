import { select } from '@databricks/design-system/test-utils/rtl';
import userEvent from '@testing-library/user-event';
import { renderWithIntl, screen } from '../../common/utils/TestUtils.react18';
import type { RunInfoEntity } from '../types';
import { CompareRunBox } from './CompareRunBox';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000); // Larger timeout for integration testing (plotly rendering)

describe('CompareRunBox', () => {
  const runUuids = ['1', '2', '3'];
  const commonProps = {
    runUuids,
    runInfos: runUuids.map(
      (runUuid) =>
        ({
          runUuid,
          experimentId: '0',
        } as RunInfoEntity),
    ),
    runDisplayNames: runUuids,
  };

  it('renders a chart with the correct x and y axes', async () => {
    const props = {
      ...commonProps,
      paramLists: [[{ key: 'param-1', value: 1 }], [{ key: 'param-2', value: 2 }], [{ key: 'param-3', value: 3 }]],
      metricLists: [[{ key: 'metric-4', value: 4 }], [{ key: 'metric-5', value: 5 }], [{ key: 'metric-6', value: 6 }]],
    };

    renderWithIntl(<CompareRunBox {...props} />);
    expect(screen.queryByText('Select parameters/metrics to plot.')).toBeInTheDocument();

    // Select x axis value
    const xAxisSelector = screen.getByRole('combobox', { name: /X-axis/ });
    await userEvent.click(xAxisSelector);
    const xOptionNames = select.getOptionNames(xAxisSelector);
    const xAxisIdx = xOptionNames.indexOf('param-3');
    await userEvent.click(select.getOptions(xAxisSelector)[xAxisIdx]);

    expect(select.getDisplayLabel(xAxisSelector)).toBe('param-3');

    // Select y axis value
    const yAxisSelector = screen.getByRole('combobox', { name: /Y-axis/ });
    await userEvent.click(yAxisSelector);
    const yOptionNames = select.getOptionNames(yAxisSelector);
    const yAxisIdx = yOptionNames.indexOf('metric-4');
    await userEvent.click(select.getOptions(yAxisSelector)[yAxisIdx]);

    expect(select.getDisplayLabel(yAxisSelector)).toBe('metric-4');

    expect(screen.queryByText('Select parameters/metrics to plot.')).not.toBeInTheDocument();
  });
});
