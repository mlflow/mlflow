import { describe, expect, it, jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { TrafficSplitConfigurator } from './TrafficSplitConfigurator';
import type { TrafficSplitModel } from '../../hooks/useEditEndpointForm';

// Mock child components to isolate unit tests
jest.mock('./TrafficSplitModelItem', () => ({
  TrafficSplitModelItem: ({ model, index }: { model: TrafficSplitModel; index: number }) => (
    <div data-testid={`traffic-split-model-${index}`}>
      {model.provider} / {model.modelName} ({model.weight}%)
    </div>
  ),
}));

const makeModel = (overrides: Partial<TrafficSplitModel> = {}): TrafficSplitModel => ({
  modelDefinitionName: '',
  provider: 'openai',
  modelName: 'gpt-4',
  secretMode: 'new',
  existingSecretId: '',
  newSecret: { name: '', authMode: '', secretFields: {}, configFields: {} },
  weight: 100,
  ...overrides,
});

describe('TrafficSplitConfigurator', () => {
  it('renders the load balancing description text', () => {
    renderWithDesignSystem(<TrafficSplitConfigurator value={[]} onChange={jest.fn()} />);
    expect(
      screen.getByText('Use multiple models for load balancing by splitting traffic with weights'),
    ).toBeInTheDocument();
  });

  it('renders the Add model button', () => {
    renderWithDesignSystem(<TrafficSplitConfigurator value={[]} onChange={jest.fn()} />);
    expect(screen.getByText('Add model')).toBeInTheDocument();
  });

  it('calls onChange with a new empty model when Add model is clicked', async () => {
    const onChange = jest.fn();
    renderWithDesignSystem(<TrafficSplitConfigurator value={[]} onChange={onChange} />);
    await userEvent.click(screen.getByText('Add model'));
    expect(onChange).toHaveBeenCalledTimes(1);
    const newModels = onChange.mock.calls[0][0] as TrafficSplitModel[];
    expect(newModels).toHaveLength(1);
    expect(newModels[0].weight).toBe(0);
    expect(newModels[0].provider).toBe('');
  });

  it('renders model items for each model in value', () => {
    const models = [makeModel({ weight: 60 }), makeModel({ provider: 'anthropic', modelName: 'claude-3', weight: 40 })];
    renderWithDesignSystem(<TrafficSplitConfigurator value={models} onChange={jest.fn()} />);
    expect(screen.getByTestId('traffic-split-model-0')).toBeInTheDocument();
    expect(screen.getByTestId('traffic-split-model-1')).toBeInTheDocument();
  });

  it('shows valid weight total when weights sum to 100', () => {
    const models = [makeModel({ weight: 60 }), makeModel({ weight: 40 })];
    renderWithDesignSystem(<TrafficSplitConfigurator value={models} onChange={jest.fn()} />);
    expect(screen.getByText(/Total:.*100%/)).toBeInTheDocument();
    expect(screen.queryByText('(must equal 100%)')).not.toBeInTheDocument();
  });

  it('shows invalid weight total when weights do not sum to 100', () => {
    const models = [makeModel({ weight: 60 }), makeModel({ weight: 20 })];
    renderWithDesignSystem(<TrafficSplitConfigurator value={models} onChange={jest.fn()} />);
    expect(screen.getByText(/Total:.*80%/)).toBeInTheDocument();
    expect(screen.getByText(/must equal 100%/)).toBeInTheDocument();
  });

  it('does not show weight total when there are no models', () => {
    renderWithDesignSystem(<TrafficSplitConfigurator value={[]} onChange={jest.fn()} />);
    expect(screen.queryByText(/Total:/)).not.toBeInTheDocument();
  });
});
