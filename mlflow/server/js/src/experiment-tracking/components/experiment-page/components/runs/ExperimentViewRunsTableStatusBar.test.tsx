import { describe, expect, jest, test } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { ExperimentViewRunsTableStatusBar } from './ExperimentViewRunsTableStatusBar';

describe('ExperimentViewRunsTableStatusBar page size', () => {
  test('offers bounded options and explains that the count includes children', async () => {
    const onChange = jest.fn();
    renderWithDesignSystem(
      <ExperimentViewRunsTableStatusBar
        allRunsCount={101}
        isLoading={false}
        pageSizeSelect={{ options: [100, 250, 500], default: 100, onChange }}
      />,
    );

    const label = screen.getByText('Runs per request (including children)');
    await userEvent.hover(label);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('parent rows may be fewer');
    await userEvent.click(screen.getByRole('combobox', { name: 'Runs per request (including children)' }));
    await waitFor(() => expect(screen.queryByRole('tooltip')).not.toBeInTheDocument());
    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['100', '250', '500']);
    await userEvent.click(screen.getByRole('option', { name: '500' }));
    expect(onChange).toHaveBeenCalledWith(500);
  });
});
