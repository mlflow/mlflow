import { describe, it, expect, jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';

import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { EvalRunsColumnsButton } from './EvalRunsColumnsButton';
import { EvalRunsTableKeyedColumnPrefix } from './ExperimentEvaluationRunsTable.constants';
import { createEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';

const metric = (key: string) => createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.METRIC, key);
const ACCURACY = metric('accuracy');
const F1 = metric('f1');

describe('EvalRunsColumnsButton', () => {
  // @ui-test-skill-generated
  it('selects every column in a group when the group is not fully selected', async () => {
    const setSelectedColumns = jest.fn();
    renderWithDesignSystem(
      <EvalRunsColumnsButton
        selectedColumns={{ [ACCURACY]: true, [F1]: false }}
        setSelectedColumns={setSelectedColumns}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Columns' }));
    // The group header ("Metrics") doubles as the show/hide-all toggle.
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Metrics' }));

    // Partial group -> the header turns the whole group on.
    expect(setSelectedColumns).toHaveBeenCalledWith({ [ACCURACY]: true, [F1]: true });
  });

  it('clears every column in a group when the group is fully selected', async () => {
    const setSelectedColumns = jest.fn();
    renderWithDesignSystem(
      <EvalRunsColumnsButton
        selectedColumns={{ [ACCURACY]: true, [F1]: true }}
        setSelectedColumns={setSelectedColumns}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Columns' }));
    const groupToggle = screen.getByRole('menuitemcheckbox', { name: 'Metrics' });
    expect(groupToggle).toHaveAttribute('aria-checked', 'true');

    await userEvent.click(groupToggle);
    expect(setSelectedColumns).toHaveBeenCalledWith({ [ACCURACY]: false, [F1]: false });
  });

  it('renders a plain header (no show/hide-all toggle) for a single-column group', async () => {
    renderWithDesignSystem(
      <EvalRunsColumnsButton selectedColumns={{ [ACCURACY]: true }} setSelectedColumns={jest.fn()} />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Columns' }));
    // Only the column itself is a checkbox; the "Metrics" header is a non-interactive label.
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Metrics' })).not.toBeInTheDocument();
    expect(screen.getByRole('menuitemcheckbox', { name: 'accuracy' })).toBeInTheDocument();
  });
});
