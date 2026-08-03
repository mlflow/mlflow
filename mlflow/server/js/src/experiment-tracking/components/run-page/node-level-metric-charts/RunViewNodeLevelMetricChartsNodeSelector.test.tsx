import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem } from '../../../../common/utils/TestUtils.react18';
import { RunViewNodeLevelMetricChartsNodeSelector } from './RunViewNodeLevelMetricChartsNodeSelector';
import { shouldEnableNodeLevelSystemMetricCharts } from '../../../../common/utils/FeatureUtils';

jest.mock('../../../../common/utils/FeatureUtils');

const defaultNodesWithGpusConfig = [
  { nodeId: 'node_0', gpuCount: 2 },
  { nodeId: 'node_1', gpuCount: 1 },
];

describe('RunViewNodeLevelMetricChartsNodeSelector', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableNodeLevelSystemMetricCharts).mockReturnValue(true);
  });

  it('renders filter button and shows node options when dropdown is opened', async () => {
    renderWithDesignSystem(
      <RunViewNodeLevelMetricChartsNodeSelector
        nodesWithGpusConfig={defaultNodesWithGpusConfig}
        onToggleGpu={jest.fn()}
        onToggleNode={jest.fn()}
        selectedNodes={new Set()}
        selectedGpus={new Map()}
      />,
    );

    const trigger = screen.getByRole('button', { name: /Filter by node/ });
    expect(trigger).toBeInTheDocument();

    await userEvent.click(trigger);

    expect(screen.getByText(/Node node_0/)).toBeInTheDocument();
    expect(screen.getByText(/Node node_1/)).toBeInTheDocument();
  });

  it('calls onToggleNode with node id when node row is clicked', async () => {
    const onToggleNode = jest.fn();
    renderWithDesignSystem(
      <RunViewNodeLevelMetricChartsNodeSelector
        nodesWithGpusConfig={defaultNodesWithGpusConfig}
        onToggleGpu={jest.fn()}
        onToggleNode={onToggleNode}
        selectedNodes={new Set()}
        selectedGpus={new Map()}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: /Filter by node/ }));
    await userEvent.click(screen.getByText(/Node node_0/));

    expect(onToggleNode).toHaveBeenCalledTimes(1);
    expect(onToggleNode).toHaveBeenCalledWith('node_0');
  });

  it('displays selection count in button when nodes are selected', () => {
    renderWithDesignSystem(
      <RunViewNodeLevelMetricChartsNodeSelector
        nodesWithGpusConfig={defaultNodesWithGpusConfig}
        onToggleGpu={jest.fn()}
        onToggleNode={jest.fn()}
        selectedNodes={new Set(['node_0'])}
        selectedGpus={new Map()}
      />,
    );

    expect(screen.getByRole('button', { name: /1 node/ })).toBeInTheDocument();
  });

  it('shows Clear filter and calls onClear when provided and clicked', async () => {
    const onClear = jest.fn();
    renderWithDesignSystem(
      <RunViewNodeLevelMetricChartsNodeSelector
        nodesWithGpusConfig={defaultNodesWithGpusConfig}
        onToggleGpu={jest.fn()}
        onToggleNode={jest.fn()}
        selectedNodes={new Set()}
        selectedGpus={new Map()}
        onClear={onClear}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: /Filter by node/ }));
    expect(screen.getByText(/Clear filter/)).toBeInTheDocument();

    await userEvent.click(screen.getByText(/Clear filter/));
    expect(onClear).toHaveBeenCalledTimes(1);
  });
});
