import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { act, renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { CREATE_NEW_MODEL_OPTION_VALUE, RegisterModelForm, SELECTED_MODEL_FIELD } from './RegisterModelForm';
import { mockRegisteredModelDetailed } from '../test-utils';

describe('RegisterModelForm', () => {
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = {
      modelByName: {},
      innerRef: React.createRef(),
      onSearchRegisteredModels: jest.fn(),
    };
  });

  const renderForm = (props = minimalProps) => renderWithDesignSystem(<RegisterModelForm {...props} />);

  const openModelSelect = async () => {
    await userEvent.click(screen.getByRole('combobox'));
  };

  test('lists "Create New Model" and existing models in the dropdown', async () => {
    renderForm({
      ...minimalProps,
      modelByName: { 'Model A': mockRegisteredModelDetailed('Model A', []) },
    });
    await openModelSelect();

    expect(screen.getByText('Create New Model')).toBeInTheDocument();
    expect(screen.getByText('Model A')).toBeInTheDocument();
  });

  test('shows the model-name input after choosing "Create New Model"', async () => {
    renderForm({
      ...minimalProps,
      modelByName: { 'Model A': mockRegisteredModelDetailed('Model A', []) },
    });
    await openModelSelect();
    await userEvent.click(screen.getByText('Create New Model'));

    expect(screen.getByLabelText('Model Name')).toBeInTheDocument();
  });

  test('submits the chosen model name through the form, including the create-new value', async () => {
    const innerRef = React.createRef<any>();
    const { rerender } = renderForm({
      ...minimalProps,
      innerRef,
      modelByName: { 'Model A': mockRegisteredModelDetailed('Model A', []) },
    });
    await openModelSelect();
    await userEvent.click(screen.getByRole('option', { name: 'Model A' }));

    await waitFor(async () => {
      expect((await innerRef.current.validateFields())[SELECTED_MODEL_FIELD]).toBe('Model A');
    });
    rerender(
      <RegisterModelForm
        {...minimalProps}
        innerRef={innerRef}
        modelByName={{ 'Model B': mockRegisteredModelDetailed('Model B', []) }}
      />,
    );
    expect(screen.getByRole('textbox', { name: 'Model' })).toHaveValue('Model A');

    await openModelSelect();
    await userEvent.click(screen.getByRole('option', { name: 'Create New Model' }));
    expect(screen.getByLabelText('Model Name')).toBeInTheDocument();
    expect(innerRef.current.getFieldValue(SELECTED_MODEL_FIELD)).toBe(CREATE_NEW_MODEL_OPTION_VALUE);
  });

  test('restores the selected model when search text loses focus', async () => {
    const innerRef = React.createRef<any>();
    renderForm({
      ...minimalProps,
      innerRef,
      modelByName: { 'Model A': mockRegisteredModelDetailed('Model A', []) },
    });
    await openModelSelect();
    await userEvent.click(screen.getByRole('option', { name: 'Model A' }));

    const modelInput = screen.getByRole('textbox', { name: 'Model' });
    await userEvent.click(modelInput);
    await userEvent.paste('Model B');
    await userEvent.tab();

    expect(modelInput).toHaveValue('Model A');
    expect(innerRef.current.getFieldValue(SELECTED_MODEL_FIELD)).toBe('Model A');
  });

  test('clears the selected model after the form is reset', async () => {
    const innerRef = React.createRef<any>();
    renderForm({
      ...minimalProps,
      innerRef,
      modelByName: { 'Model A': mockRegisteredModelDetailed('Model A', []) },
    });
    await openModelSelect();
    await userEvent.click(screen.getByRole('option', { name: 'Model A' }));

    act(() => innerRef.current.resetFields());

    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Model' })).toHaveValue(''));
  });

  test('searches registered models as the user types a model name', async () => {
    const onSearchRegisteredModels = jest.fn();
    renderForm({
      ...minimalProps,
      modelByName: { 'Model A': mockRegisteredModelDetailed('Model A', []) },
      onSearchRegisteredModels,
    });
    await openModelSelect();
    await userEvent.click(screen.getByRole('textbox', { name: 'Model' }));
    await userEvent.paste('Model B');

    expect(onSearchRegisteredModels).toHaveBeenLastCalledWith('Model B');
  });

  test('filters model options by search text', async () => {
    const { rerender } = renderForm({
      ...minimalProps,
      modelByName: {
        'Model A': mockRegisteredModelDetailed('Model A', []),
        'Model B': mockRegisteredModelDetailed('Model B', []),
      },
    });
    await openModelSelect();

    const modelInput = screen.getByRole('textbox', { name: 'Model' });
    await userEvent.paste('Model B');

    expect(screen.getByRole('option', { name: 'Model B' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Model A' })).not.toBeInTheDocument();

    rerender(
      <RegisterModelForm
        {...minimalProps}
        modelByName={{
          'Model B': mockRegisteredModelDetailed('Model B', []),
          'Model C': mockRegisteredModelDetailed('Model C', []),
        }}
      />,
    );
    await waitFor(() => {
      expect(screen.getByRole('option', { name: 'Model B' })).toBeInTheDocument();
      expect(screen.queryByRole('option', { name: 'Model C' })).not.toBeInTheDocument();
    });

    await userEvent.clear(modelInput);
    await userEvent.paste('No match');

    expect(screen.queryByText('Models')).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Model B' })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Create New Model' })).not.toBeInTheDocument();
  });

  test('uses the copy form label as the combobox accessible name', () => {
    renderForm({ ...minimalProps, isCopy: true });

    expect(screen.getByRole('combobox', { name: 'Copy to model' })).toBeInTheDocument();
  });
});
