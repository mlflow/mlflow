import { describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEventGlobal, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ResponseFormatSchemaBuilder } from './ResponseFormatSchemaBuilder';

// SimpleSelect renders as a radix combobox, which fails jsdom's pointer-events
// check when clicked; disable it, matching the convention used for other
// radix-based SimpleSelect interactions in this repo.
const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const singleFieldSchema = JSON.stringify({
  type: 'object',
  properties: { status: { type: 'string' } },
  required: [],
});

const twoFieldSchema = JSON.stringify({
  type: 'object',
  properties: { name: { type: 'string' }, age: { type: 'integer' } },
  required: [],
});

const enumFieldSchema = JSON.stringify({
  type: 'object',
  properties: { priority: { type: 'string', enum: ['low', 'medium', 'high'] } },
  required: [],
});

const emptyEnumFieldSchema = JSON.stringify({
  type: 'object',
  properties: { priority: { type: 'string', enum: [] } },
  required: [],
});

describe('ResponseFormatSchemaBuilder', () => {
  const renderComponent = (props: { schemaText: string; onSchemaChange: (next: string) => void }) => {
    return render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ResponseFormatSchemaBuilder {...props} />
        </DesignSystemProvider>
      </IntlProvider>,
    );
  };

  it('renders fields parsed from the given schemaText', () => {
    renderComponent({ schemaText: singleFieldSchema, onSchemaChange: jest.fn<(next: string) => void>() });
    expect(screen.getByDisplayValue('status')).toBeInTheDocument();
  });

  it('adds an empty property row when "Add property" is clicked', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: '', onSchemaChange });

    await userEvent.click(screen.getByRole('button', { name: 'Add property' }));

    expect(screen.getAllByPlaceholderText('Property')).toHaveLength(1);
    expect(onSchemaChange).toHaveBeenCalledTimes(1);
    const emitted = JSON.parse(onSchemaChange.mock.calls[0][0]);
    expect(emitted.properties).toEqual({});
  });

  it('removes a property and emits a schema without it when its delete button is clicked', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: twoFieldSchema, onSchemaChange });

    const nameInputs = screen.getAllByRole('textbox');
    expect(nameInputs.map((input) => (input as HTMLInputElement).value)).toEqual(['name', 'age']);

    const deleteButtons = screen.getAllByRole('button', { name: 'Remove property' });
    await userEvent.click(deleteButtons[1]); // remove "age"

    expect(screen.getAllByRole('textbox')).toHaveLength(1);
    const emitted = JSON.parse(onSchemaChange.mock.calls[onSchemaChange.mock.calls.length - 1][0]);
    expect(emitted.properties).toEqual({ name: { type: 'string' } });
  });

  it('shows the enum value controls only when the type is switched to enum', async () => {
    renderComponent({ schemaText: singleFieldSchema, onSchemaChange: jest.fn<(next: string) => void>() });

    expect(screen.queryByRole('button', { name: 'Add enum value' })).not.toBeInTheDocument();

    const trigger = document.querySelector<HTMLElement>(
      '[data-component-id="mlflow.prompts.create.response_format.property_type"]',
    );
    if (!trigger) throw new Error('Property type SimpleSelect trigger not found');
    await userEvent.click(trigger);
    await userEvent.click(await screen.findByRole('option', { name: 'enum' }));

    expect(screen.getByRole('button', { name: 'Add enum value' })).toBeInTheDocument();

    await userEvent.click(trigger);
    await userEvent.click(await screen.findByRole('option', { name: 'string' }));

    expect(screen.queryByRole('button', { name: 'Add enum value' })).not.toBeInTheDocument();
  });

  it('renders existing enum values as individual, editable rows', () => {
    renderComponent({ schemaText: enumFieldSchema, onSchemaChange: jest.fn<(next: string) => void>() });

    expect(screen.getByDisplayValue('low')).toBeInTheDocument();
    expect(screen.getByDisplayValue('medium')).toBeInTheDocument();
    expect(screen.getByDisplayValue('high')).toBeInTheDocument();
  });

  it('adds and populates a new enum value row', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: emptyEnumFieldSchema, onSchemaChange });

    await userEvent.click(screen.getByRole('button', { name: 'Add enum value' }));
    await userEvent.type(screen.getByPlaceholderText('Allowed value'), 'low');

    const emitted = JSON.parse(onSchemaChange.mock.calls[onSchemaChange.mock.calls.length - 1][0]);
    expect(emitted.properties.priority.enum).toEqual(['low']);
  });

  it('removes a single enum value without affecting the others', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: enumFieldSchema, onSchemaChange });

    const removeButtons = screen.getAllByRole('button', { name: 'Remove enum value' });
    await userEvent.click(removeButtons[1]); // remove "medium"

    expect(screen.getByDisplayValue('low')).toBeInTheDocument();
    expect(screen.queryByDisplayValue('medium')).not.toBeInTheDocument();
    expect(screen.getByDisplayValue('high')).toBeInTheDocument();

    const emitted = JSON.parse(onSchemaChange.mock.calls[onSchemaChange.mock.calls.length - 1][0]);
    expect(emitted.properties.priority.enum).toEqual(['low', 'high']);
  });

  it('emits an array schema when the array toggle is switched on', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: singleFieldSchema, onSchemaChange });

    // Row order is [array toggle, required toggle]
    const [arraySwitch] = screen.getAllByRole('switch');
    await userEvent.click(arraySwitch);

    const emitted = JSON.parse(onSchemaChange.mock.calls[onSchemaChange.mock.calls.length - 1][0]);
    expect(emitted.properties.status).toEqual({ type: 'array', items: { type: 'string' } });
  });

  it('emits a schema with the field in `required` when the required toggle is switched on', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: singleFieldSchema, onSchemaChange });

    // Row order is [array toggle, required toggle]
    const [, requiredSwitch] = screen.getAllByRole('switch');
    await userEvent.click(requiredSwitch);

    const emitted = JSON.parse(onSchemaChange.mock.calls[onSchemaChange.mock.calls.length - 1][0]);
    expect(emitted.required).toEqual(['status']);
  });
});
