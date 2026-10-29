import { describe, expect, test, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { slowlyTypeEachKey } from '../test-utils/slowlyTypeEachKey';
import { TraceFilterButton } from './TraceFilterButton';
import { FilterOp, type FilterFieldDef } from './filterModel';

const FIELDS: FilterFieldDef[] = [
  {
    id: 'session',
    label: 'Session',
    operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
    valueInput: 'combobox',
    options: [
      { value: 'session-1', label: 'session-1' },
      { value: 'session-2', label: 'session-2' },
    ],
  },
  {
    id: 'tag',
    label: 'Tag',
    operators: [FilterOp.EQUALS],
    valueInput: 'combobox',
    requiresKey: true,
    keyInput: 'combobox',
    keyOptions: [
      { value: 'env', label: 'env' },
      { value: 'team', label: 'team' },
    ],
    getValueOptions: (key) =>
      key === 'env'
        ? [
            { value: 'prod', label: 'prod' },
            { value: 'staging', label: 'staging' },
          ]
        : [{ value: 'ml', label: 'ml' }],
  },
];

const renderButton = (onChange = jest.fn()) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TraceFilterButton
          fields={FIELDS}
          filterModel={[]}
          onChange={onChange}
          onClearAll={jest.fn()}
          activeCount={0}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('TraceFilterButton combobox value input', () => {
  test('suggests the field options and applies a picked one', async () => {
    const onChange = jest.fn();
    renderButton(onChange);
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));

    const valueCombobox = await screen.findByRole('combobox', { name: 'Filter value' });
    expect(screen.queryByRole('textbox', { name: /Filter value/ })).not.toBeInTheDocument();
    await userEvent.click(valueCombobox);
    expect(await screen.findByRole('option', { name: 'session-1' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'session-2' }));
    await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }));

    expect(onChange).toHaveBeenCalledWith([{ field: 'session', operator: FilterOp.EQUALS, value: 'session-2' }]);
  });

  test('accepts a typed value that is not suggested', async () => {
    const onChange = jest.fn();
    renderButton(onChange);
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));

    await userEvent.click(await screen.findByRole('combobox', { name: 'Filter value' }));
    await slowlyTypeEachKey(screen.getByRole('searchbox'), 'session-9');
    await userEvent.click(await screen.findByRole('option', { name: 'Use "session-9"' }));
    await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }));

    expect(onChange).toHaveBeenCalledWith([{ field: 'session', operator: FilterOp.EQUALS, value: 'session-9' }]);
  });

  test('suggests values for the chosen key', async () => {
    const onChange = jest.fn();
    renderButton(onChange);
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    await userEvent.click(await screen.findByRole('combobox', { name: /Filter field/ }));
    await userEvent.click(await screen.findByRole('option', { name: 'Tag' }));

    await userEvent.click(await screen.findByRole('combobox', { name: /Filter key/ }));
    await userEvent.click(await screen.findByRole('option', { name: 'env' }));
    await userEvent.click(screen.getByRole('combobox', { name: 'Filter value' }));
    expect(await screen.findByRole('option', { name: 'staging' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'ml' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'prod' }));
    await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }));

    expect(onChange).toHaveBeenCalledWith([{ field: 'tag', operator: FilterOp.EQUALS, value: 'prod', key: 'env' }]);
  });
});
