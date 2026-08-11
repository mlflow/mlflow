import { describe, expect, test, jest, beforeEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { slowlyTypeEachKey } from '../test-utils/slowlyTypeEachKey';
import { TraceFilterButton } from './TraceFilterButton';
import { FilterOp, type FilterFieldDef, type TraceFilterModel } from './filterModel';

const FIELDS: FilterFieldDef[] = [
  {
    id: 'state',
    label: 'State',
    operators: [FilterOp.EQUALS],
    valueInput: 'select',
    options: [
      { value: 'OK', label: 'OK' },
      { value: 'ERROR', label: 'Error' },
    ],
  },
  {
    id: 'duration',
    label: 'Duration',
    operators: [FilterOp.GREATER_THAN, FilterOp.LESS_THAN],
    valueInput: 'number',
  },
  {
    id: 'tag',
    label: 'Tag',
    operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
    valueInput: 'text',
    requiresKey: true,
  },
  {
    id: 'assessment',
    label: 'Assessment',
    operators: [FilterOp.EQUALS, FilterOp.NOT_EQUALS],
    valueInput: 'text',
    requiresKey: true,
    keyInput: 'combobox',
    keyOptions: [
      { value: 'relevance', label: 'relevance' },
      { value: 'safety', label: 'safety' },
    ],
  },
];

const renderButton = (over: Partial<React.ComponentProps<typeof TraceFilterButton>> = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TraceFilterButton
          fields={FIELDS}
          filterModel={[]}
          onChange={jest.fn()}
          onClearAll={jest.fn()}
          activeCount={0}
          {...over}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('TraceFilterButton', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('opens a popover with one blank clause seeded from the first field', async () => {
    renderButton();
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    // The field select shows the first field's label as its value.
    expect(await screen.findByRole('combobox', { name: /Filter field/ })).toHaveTextContent('State');
  });

  test('adding a clause renders a second row', async () => {
    renderButton();
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    await userEvent.click(await screen.findByRole('button', { name: 'Add filter' }));
    expect(screen.getAllByRole('combobox', { name: /Filter field/ })).toHaveLength(2);
  });

  test('removing a clause collapses back to a single blank row', async () => {
    renderButton();
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    await userEvent.click(await screen.findByRole('button', { name: 'Add filter' }));
    const removeButtons = screen.getAllByRole('button', { name: 'Remove filter' });
    await userEvent.click(removeButtons[0]);
    expect(screen.getAllByRole('combobox', { name: /Filter field/ })).toHaveLength(1);
  });

  test('the value input is a select for a select-kind field', async () => {
    renderButton();
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    // The default field (state) is select-kind → a value selector, not a text input.
    expect(await screen.findByRole('combobox', { name: /Filter value/ })).toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: /Filter value/ })).not.toBeInTheDocument();
  });

  test('Apply commits the draft via onChange', async () => {
    const onChange = jest.fn();
    renderButton({ onChange });
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
    // Pick a state value, then apply.
    await userEvent.click(await screen.findByRole('combobox', { name: /Filter value/ }));
    await userEvent.click(await screen.findByRole('option', { name: 'Error' }));
    await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }));
    expect(onChange).toHaveBeenCalledWith([{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }]);
  });

  test('a requiresKey field renders the Key input and Apply emits a clause carrying the key', async () => {
    const onChange = jest.fn();
    renderButton({ onChange });
    await userEvent.click(screen.getByRole('button', { name: /Filters/ }));

    // Switch the first row to the key-requiring Tag field.
    await userEvent.click(await screen.findByRole('combobox', { name: /Filter field/ }));
    await userEvent.click(await screen.findByRole('option', { name: 'Tag' }));

    // The Key input appears only for a requiresKey field; fill both key and value, then apply.
    await userEvent.click(await screen.findByRole('textbox', { name: /Filter key/ }));
    await userEvent.paste('env');
    await userEvent.click(screen.getByRole('textbox', { name: /Filter value/ }));
    await userEvent.paste('prod');
    await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }));

    expect(onChange).toHaveBeenCalledWith([{ field: 'tag', operator: FilterOp.EQUALS, value: 'prod', key: 'env' }]);
  });

  describe('combobox key input (keyInput: combobox)', () => {
    // Switch the first row to the Assessment field, whose key renders as a freeform combobox.
    const selectAssessmentField = async () => {
      await userEvent.click(screen.getByRole('button', { name: /Filters/ }));
      await userEvent.click(await screen.findByRole('combobox', { name: /Filter field/ }));
      await userEvent.click(await screen.findByRole('option', { name: 'Assessment' }));
    };

    test('renders a combobox (not a plain text input) for the key and lists the keyOptions', async () => {
      renderButton();
      await selectAssessmentField();

      // The key is a combobox, not the plain text input the tag/metadata fields use.
      const keyCombobox = await screen.findByRole('combobox', { name: /Filter key/ });
      expect(keyCombobox).toBeInTheDocument();
      expect(screen.queryByRole('textbox', { name: /Filter key/ })).not.toBeInTheDocument();

      // Opening it lists the suggested key options.
      await userEvent.click(keyCombobox);
      expect(await screen.findByRole('option', { name: 'relevance' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'safety' })).toBeInTheDocument();
    });

    test('selecting a suggested key and applying emits a clause carrying that key', async () => {
      const onChange = jest.fn();
      renderButton({ onChange });
      await selectAssessmentField();

      await userEvent.click(await screen.findByRole('combobox', { name: /Filter key/ }));
      await userEvent.click(await screen.findByRole('option', { name: 'relevance' }));
      await userEvent.click(screen.getByRole('textbox', { name: /Filter value/ }));
      await userEvent.paste('yes');
      await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }));

      expect(onChange).toHaveBeenCalledWith([
        { field: 'assessment', operator: FilterOp.EQUALS, value: 'yes', key: 'relevance' },
      ]);
    });

    test('typing a name not in the options offers a Use "<typed>" item that sets the key', async () => {
      const onChange = jest.fn();
      renderButton({ onChange });
      await selectAssessmentField();

      await userEvent.click(await screen.findByRole('combobox', { name: /Filter key/ }));
      // The combobox search box accepts a freeform name absent from the suggestions. Per-keystroke
      // typing drives the typeahead, so use the sanctioned slow-type helper.
      await slowlyTypeEachKey(screen.getByRole('searchbox'), 'custom_judge');
      await userEvent.click(await screen.findByRole('option', { name: 'Use "custom_judge"' }));

      await userEvent.click(screen.getByRole('textbox', { name: /Filter value/ }));
      await userEvent.paste('5');
      await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }));

      expect(onChange).toHaveBeenCalledWith([
        { field: 'assessment', operator: FilterOp.EQUALS, value: '5', key: 'custom_judge' },
      ]);
    });
  });

  test('shows an active count and clear-all when filters are applied', async () => {
    const onClearAll = jest.fn();
    const filterModel: TraceFilterModel = [{ field: 'state', operator: FilterOp.EQUALS, value: 'OK' }];
    renderButton({ filterModel, activeCount: 1, onClearAll });
    // The trigger shows the (1) count.
    expect(screen.getByRole('button', { name: /Filters/ })).toHaveTextContent('(1)');
    // The sibling clear-all button resets everything.
    await userEvent.click(screen.getByRole('button', { name: 'Clear all filters' }));
    expect(onClearAll).toHaveBeenCalled();
  });
});
