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

const richSchema = JSON.stringify({
  type: 'object',
  additionalProperties: false,
  properties: {
    summary: { type: 'string', description: 'One line summary' },
    owner: { $ref: '#/$defs/Person' },
  },
  $defs: { Person: { type: 'object', properties: { name: { type: 'string' } } } },
});

const emptyEnumFieldSchema = JSON.stringify({
  type: 'object',
  properties: { priority: { type: 'string', enum: [] } },
  required: [],
});

const lastEmitted = (onSchemaChange: jest.Mock<(next: string) => void>) =>
  JSON.parse(onSchemaChange.mock.lastCall?.[0] ?? '');

// Each row renders its switches in the order [array toggle, required toggle]
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
    const emitted = lastEmitted(onSchemaChange);
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
    const emitted = lastEmitted(onSchemaChange);
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

    const emitted = lastEmitted(onSchemaChange);
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

    const emitted = lastEmitted(onSchemaChange);
    expect(emitted.properties.priority.enum).toEqual(['low', 'high']);
  });

  it('emits an array schema when the array toggle is switched on', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: singleFieldSchema, onSchemaChange });

    const [arraySwitch] = screen.getAllByRole('switch');
    await userEvent.click(arraySwitch);

    const emitted = lastEmitted(onSchemaChange);
    expect(emitted.properties.status).toEqual({ type: 'array', items: { type: 'string' } });
  });

  it('emits a schema with the field in `required` when the required toggle is switched on', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: singleFieldSchema, onSchemaChange });

    const [, requiredSwitch] = screen.getAllByRole('switch');
    await userEvent.click(requiredSwitch);

    const emitted = lastEmitted(onSchemaChange);
    expect(emitted.required).toEqual(['status']);
  });

  it('keeps keywords it does not model when a field is edited', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: richSchema, onSchemaChange });

    const [arraySwitch] = screen.getAllByRole('switch');
    await userEvent.click(arraySwitch);

    const emitted = lastEmitted(onSchemaChange);
    expect(emitted.additionalProperties).toBe(false);
    expect(emitted.properties.summary.description).toEqual('One line summary');
    expect(emitted.properties.owner).toEqual({ $ref: '#/$defs/Person' });
  });

  it('shows fields it cannot represent as read-only rows', () => {
    renderComponent({ schemaText: richSchema, onSchemaChange: jest.fn<(next: string) => void>() });

    const ownerNameInput = screen.getByDisplayValue('owner');
    expect(ownerNameInput).toBeDisabled();
    expect(screen.getByDisplayValue('$ref')).toBeInTheDocument();
    expect(screen.getByDisplayValue('summary')).toBeEnabled();
  });

  it('flags a duplicate property name', async () => {
    renderComponent({ schemaText: twoFieldSchema, onSchemaChange: jest.fn<(next: string) => void>() });

    const ageInput = screen.getByDisplayValue('age');
    await userEvent.clear(ageInput);
    await userEvent.type(ageInput, 'name');

    expect(screen.getAllByText('Duplicate property name. Only the last field with this name is saved.')).toHaveLength(
      2,
    );
  });

  it('warns that an unnamed property is not saved', async () => {
    renderComponent({ schemaText: '', onSchemaChange: jest.fn<(next: string) => void>() });

    await userEvent.click(screen.getByRole('button', { name: 'Add property' }));

    expect(screen.getByText('Properties without a name are not saved.')).toBeInTheDocument();
  });

  it('emits an empty value when the last property is removed, so no schema is saved', async () => {
    const onSchemaChange = jest.fn<(next: string) => void>();
    renderComponent({ schemaText: singleFieldSchema, onSchemaChange });

    await userEvent.click(screen.getByRole('button', { name: 'Remove property' }));

    expect(onSchemaChange).toHaveBeenLastCalledWith('');
  });
});
