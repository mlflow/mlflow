import { describe, expect, jest, beforeEach, test } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { useModelsQuery } from '../../hooks/useModelsQuery';
import { ModelSelectorModal } from './ModelSelectorModal';

jest.mock('../../hooks/useModelsQuery', () => ({
  useModelsQuery: jest.fn(),
}));

describe('ModelSelectorModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useModelsQuery).mockReturnValue({
      data: [
        { model: 'gpt-5.6-sol', provider: 'openai', supports_function_calling: true },
        { model: 'gpt-5', provider: 'openai', supports_function_calling: true },
        { model: 'gpt-5-mini', provider: 'openai', supports_function_calling: true },
        { model: 'gpt-4.1-nano', provider: 'openai', supports_function_calling: true },
      ],
      isLoading: false,
      refetch: jest.fn(),
    } as any);
  });

  test('defaults to the curated MLflow model list without full-catalog controls', () => {
    renderWithDesignSystem(<ModelSelectorModal isOpen onClose={jest.fn()} onSelect={jest.fn()} provider="openai" />);

    expect(screen.getByText('gpt-5.6-sol')).toBeInTheDocument();
    expect(screen.getByText('gpt-5')).toBeInTheDocument();
    expect(screen.getByText('gpt-5-mini')).toBeInTheDocument();
    expect(screen.queryByText('gpt-4.1-nano')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Search')).not.toBeInTheDocument();
    expect(screen.queryByText('Use a custom model name')).not.toBeInTheDocument();
    expect(screen.getByText('3 models available')).toBeInTheDocument();
  });

  test('full mode exposes the provider catalog controls for LLM Connections', () => {
    renderWithDesignSystem(
      <ModelSelectorModal
        isOpen
        onClose={jest.fn()}
        onSelectMultiple={jest.fn()}
        provider="openai"
        multiSelect
        modelListMode="full"
      />,
    );

    expect(screen.getByText('gpt-4.1-nano')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search')).toBeInTheDocument();
    expect(screen.getByText('Capability')).toBeInTheDocument();
    expect(screen.getByText('4 models available')).toBeInTheDocument();
  });

  test('selects a curated default model with catalog metadata when available', async () => {
    const onSelect = jest.fn();
    renderWithDesignSystem(<ModelSelectorModal isOpen onClose={jest.fn()} onSelect={onSelect} provider="openai" />);

    await userEvent.click(screen.getByText('gpt-5'));
    await userEvent.click(screen.getByText('Select'));

    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'gpt-5', provider: 'openai', supports_function_calling: true }),
    );
  });

  test('re-seeds multi-select state when initialSelected changes while open', async () => {
    const onSelectMultiple = jest.fn();
    const { rerender } = renderWithDesignSystem(
      <ModelSelectorModal
        isOpen
        onClose={jest.fn()}
        onSelectMultiple={onSelectMultiple}
        provider="openai"
        multiSelect
        modelListMode="full"
        initialSelected={[{ model: 'gpt-5', provider: 'openai', supports_function_calling: true }]}
      />,
    );

    expect(screen.getByText('Select (1)')).toBeInTheDocument();

    rerender(
      <ModelSelectorModal
        isOpen
        onClose={jest.fn()}
        onSelectMultiple={onSelectMultiple}
        provider="openai"
        multiSelect
        modelListMode="full"
        initialSelected={[{ model: 'gpt-5-mini', provider: 'openai', supports_function_calling: true }]}
      />,
    );

    expect(screen.getByText('Select (1)')).toBeInTheDocument();
    await userEvent.click(screen.getByText('Select (1)'));

    expect(onSelectMultiple).toHaveBeenCalledWith([
      expect.objectContaining({ model: 'gpt-5-mini', provider: 'openai' }),
    ]);
  });
});
