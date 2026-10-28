import { describe, expect, jest, test } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem } from '../../../../common/utils/TestUtils.react18';
import type { GenAIOverviewChartContext } from '../genAIOverview.types';
import { GenAIOverviewPromptBox } from './GenAIOverviewPromptBox';

const createSubmit = () =>
  jest.fn<(prompt: string, context?: GenAIOverviewChartContext) => Promise<'submitted' | 'unavailable' | 'failed'>>();

describe('GenAIOverviewPromptBox', () => {
  test('preserves multiline input until submission', async () => {
    const onSubmit = createSubmit().mockResolvedValue('submitted');
    renderWithDesignSystem(<GenAIOverviewPromptBox onClearChartContext={jest.fn()} onSubmit={onSubmit} />);

    const overviewInput = screen.getByRole('textbox', { name: 'Ask MLflow about this agent' });
    await userEvent.click(overviewInput);
    await userEvent.type(overviewInput, 'Line one{shift>}{enter}{/shift}Line two');

    expect(overviewInput).toHaveValue('Line one\nLine two');
    expect(onSubmit).not.toHaveBeenCalled();

    await userEvent.type(overviewInput, '{enter}');

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('Line one\nLine two', undefined));
  });

  test('grows to eight rows and then scrolls', async () => {
    renderWithDesignSystem(<GenAIOverviewPromptBox onClearChartContext={jest.fn()} onSubmit={createSubmit()} />);

    const overviewInput = screen.getByRole<HTMLTextAreaElement>('textbox', {
      name: 'Ask MLflow about this agent',
    });
    const rowHeight = Number.parseFloat(overviewInput.style.height);
    let scrollHeight = rowHeight * 3;
    Object.defineProperty(overviewInput, 'scrollHeight', {
      configurable: true,
      get: () => scrollHeight,
    });

    await userEvent.type(overviewInput, 'Three rows');
    await waitFor(() => expect(overviewInput.style.height).toBe(`${rowHeight * 3}px`));
    expect(overviewInput.style.overflowY).toBe('hidden');

    scrollHeight = rowHeight * 12;
    await userEvent.type(overviewInput, ' and more');
    await waitFor(() => expect(overviewInput.style.height).toBe(`${rowHeight * 8}px`));
    expect(overviewInput.style.overflowY).toBe('auto');
  });
});
