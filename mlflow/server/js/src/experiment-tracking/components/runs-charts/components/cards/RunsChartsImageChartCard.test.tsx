import { describe, expect, test } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { RunsChartsImageChartCard } from './RunsChartsImageChartCard';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { ExperimentRunsChartsUIConfiguration } from '../../../experiment-page/models/ExperimentPageUIState';
import { createExperimentPageUIState } from '../../../experiment-page/models/ExperimentPageUIState';
import {
  RunsChartsUIConfigurationContextProvider,
  useConfirmChartCardConfigurationFn,
} from '../../hooks/useRunsChartsUIConfiguration';
import { RunsChartsImageCardConfig } from '../../runs-charts.types';
import { LOG_IMAGE_TAG_INDICATOR } from '@mlflow/mlflow/src/experiment-tracking/constants';

const IMAGE_KEY = 'sample_output';

const buildImageCardConfig = (): RunsChartsImageCardConfig => ({
  ...new RunsChartsImageCardConfig(false, 'image-card-1'),
  imageKeys: [IMAGE_KEY],
});

const buildRunData = (): RunsChartsRunData =>
  ({
    uuid: 'run-1',
    displayName: 'training-run',
    metrics: {},
    params: { lr: { key: 'lr', value: 0.1 } },
    tags: { [LOG_IMAGE_TAG_INDICATOR]: { key: LOG_IMAGE_TAG_INDICATOR, value: 'true' } },
    images: {
      [IMAGE_KEY]: {
        sample_output_step_0: { filepath: 'a.png', step: 0, key: IMAGE_KEY } as any,
      },
    },
  }) as unknown as RunsChartsRunData;

describe('RunsChartsImageChartCard', () => {
  // Renders the card the way it appears on the runs page, alongside a helper button that
  // flips `showRunParams` through the same global-config path the configuration modal uses.
  const renderCard = () => {
    const TestComponent = () => {
      const [uiState, setUIState] = useState<ExperimentRunsChartsUIConfiguration>({
        ...createExperimentPageUIState(),
        compareRunCharts: [buildImageCardConfig()],
      });

      const config = uiState.compareRunCharts?.[0] as RunsChartsImageCardConfig;

      return (
        <RunsChartsUIConfigurationContextProvider updateChartsUIState={setUIState}>
          <RunsChartsImageChartCard
            config={config}
            chartRunData={[buildRunData()]}
            onDelete={() => {}}
            onEdit={() => {}}
            groupBy={null}
            onReorderWith={() => {}}
            canMoveUp={false}
            canMoveDown={false}
            canMoveToTop={false}
            canMoveToBottom={false}
          />
          <HideRunParamsButton config={config} />
        </RunsChartsUIConfigurationContextProvider>
      );
    };

    render(<TestComponent />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <DesignSystemProvider>{children}</DesignSystemProvider>
        </IntlProvider>
      ),
    });
  };

  test('reflects showRunParams changes on the runs page without a remount', async () => {
    renderCard();

    // Params are shown by default (showRunParams defaults to true).
    await waitFor(() => {
      expect(screen.getByText('lr=0.1')).toBeInTheDocument();
    });

    // Turning the setting off commits a new config to global state, exactly as saving the
    // configuration modal does. The card should reflect it immediately; without the tmpConfig
    // sync it would stay stale until a page refresh remounted the card.
    await userEvent.click(screen.getByText('Hide run params'));

    await waitFor(() => {
      expect(screen.queryByText('lr=0.1')).not.toBeInTheDocument();
    });
    expect(screen.getByText('training-run')).toBeInTheDocument();
  });
});

const HideRunParamsButton = ({ config }: { config: RunsChartsImageCardConfig }) => {
  const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();
  return (
    <button
      type="button"
      onClick={() => confirmChartCardConfiguration({ ...config, showRunParams: false } as RunsChartsImageCardConfig)}
    >
      Hide run params
    </button>
  );
};
