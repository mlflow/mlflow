import { describe, expect, jest, test } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { ModelSelectorModal } from './ModelSelectorModal';
import { useModelsQuery } from '../../hooks/useModelsQuery';
import { sortModelsByDate } from '../../utils/formatters';

jest.mock('../../hooks/useModelsQuery');
jest.mock('../../utils/formatters', () => ({
  ...jest.requireActual<typeof import('../../utils/formatters')>('../../utils/formatters'),
  sortModelsByDate: jest.fn(
    jest.requireActual<typeof import('../../utils/formatters')>('../../utils/formatters').sortModelsByDate,
  ),
}));

describe('ModelSelectorModal', () => {
  test('reuses the sorted catalog while searching and preserves a selected model', async () => {
    const models = [
      { provider: 'openai', model: 'model-1', supports_function_calling: false },
      { provider: 'openai', model: 'model-2', supports_function_calling: true },
    ];
    jest.mocked(useModelsQuery).mockReturnValue({ data: models, isLoading: false } as any);
    const onSelect = jest.fn();
    renderWithDesignSystem(<ModelSelectorModal isOpen provider="openai" onSelect={onSelect} onClose={jest.fn()} />);
    await userEvent.click(screen.getByText('model-2', { exact: true }));
    jest.mocked(sortModelsByDate).mockClear();
    await userEvent.type(screen.getByPlaceholderText('Search'), 'model-1');
    expect(screen.queryByText('model-2', { exact: true })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Select' }));
    expect(onSelect).toHaveBeenCalledWith(models[1]);
    expect(sortModelsByDate).not.toHaveBeenCalled();
  });
});
