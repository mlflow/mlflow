import { describe, expect, it, jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { FallbackModelsConfigurator } from './FallbackModelsConfigurator';
import type { FallbackModel } from '../../hooks/useEditEndpointForm';

// Mock drag-and-drop provider to avoid HTML5 backend issues in tests
jest.mock('react-dnd', () => ({
  DndProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useDrag: () => [{ isDragging: false }, jest.fn(), jest.fn()],
  useDrop: () => [{ isOver: false }, jest.fn()],
}));

// Mock child component to isolate unit tests
jest.mock('./FallbackModelItem', () => ({
  FallbackModelItem: ({ model, index }: { model: FallbackModel; index: number }) => (
    <div data-testid={`fallback-model-${index}`}>
      Fallback Model {model.fallbackOrder}: {model.provider} / {model.modelName}
    </div>
  ),
}));

const makeModel = (overrides: Partial<FallbackModel> = {}): FallbackModel => ({
  modelDefinitionName: '',
  provider: 'openai',
  modelName: 'gpt-4',
  secretMode: 'new',
  existingSecretId: '',
  newSecret: { name: '', authMode: '', secretFields: {}, configFields: {} },
  fallbackOrder: 1,
  ...overrides,
});

describe('FallbackModelsConfigurator', () => {
  it('renders the Add fallback button', () => {
    renderWithDesignSystem(<FallbackModelsConfigurator value={[]} onChange={jest.fn()} />);
    expect(screen.getByText('Add fallback')).toBeInTheDocument();
  });

  it('calls onChange with a new model when Add fallback is clicked with empty list', async () => {
    const onChange = jest.fn();
    renderWithDesignSystem(<FallbackModelsConfigurator value={[]} onChange={onChange} />);
    await userEvent.click(screen.getByText('Add fallback'));
    expect(onChange).toHaveBeenCalledTimes(1);
    const newModels = onChange.mock.calls[0][0] as FallbackModel[];
    expect(newModels).toHaveLength(1);
    expect(newModels[0].fallbackOrder).toBe(1);
    expect(newModels[0].provider).toBe('');
  });

  it('assigns correct fallbackOrder when adding to existing models', async () => {
    const onChange = jest.fn();
    const models = [makeModel({ fallbackOrder: 1 }), makeModel({ fallbackOrder: 2 })];
    renderWithDesignSystem(<FallbackModelsConfigurator value={models} onChange={onChange} />);
    await userEvent.click(screen.getByText('Add fallback'));
    const newModels = onChange.mock.calls[0][0] as FallbackModel[];
    expect(newModels).toHaveLength(3);
    expect(newModels[2].fallbackOrder).toBe(3);
  });

  it('renders model items for each model', () => {
    const models = [makeModel({ fallbackOrder: 1 }), makeModel({ fallbackOrder: 2, provider: 'anthropic' })];
    renderWithDesignSystem(<FallbackModelsConfigurator value={models} onChange={jest.fn()} />);
    expect(screen.getByTestId('fallback-model-0')).toBeInTheDocument();
    expect(screen.getByTestId('fallback-model-1')).toBeInTheDocument();
  });

  it('renders Fallback connector labels between models and at the top', () => {
    const models = [makeModel({ fallbackOrder: 1 }), makeModel({ fallbackOrder: 2 })];
    renderWithDesignSystem(<FallbackModelsConfigurator value={models} onChange={jest.fn()} />);
    // Top connector + between-items connector
    expect(screen.getAllByText('Fallback').length).toBeGreaterThanOrEqual(2);
  });

  it('renders top Fallback connector label with single model', () => {
    const models = [makeModel({ fallbackOrder: 1 })];
    renderWithDesignSystem(<FallbackModelsConfigurator value={models} onChange={jest.fn()} />);
    // Top connector only (no between-items connector with single model)
    expect(screen.getAllByText('Fallback')).toHaveLength(1);
  });

  it('does not render Fallback label when no models exist', () => {
    renderWithDesignSystem(<FallbackModelsConfigurator value={[]} onChange={jest.fn()} />);
    expect(screen.queryAllByText('Fallback')).toHaveLength(0);
  });
});
