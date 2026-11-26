import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { CREATE_NEW_MODEL_OPTION_VALUE, RegisterModelForm, SELECTED_MODEL_FIELD } from './RegisterModelForm';
import { mockRegisteredModelDetailed } from '../test-utils';

const mockModel = mockRegisteredModelDetailed as (
  name: string,
  latestVersions?: unknown[],
  tags?: unknown[],
) => ReturnType<typeof mockRegisteredModelDetailed>;

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
      modelByName: { 'Model A': mockModel('Model A', []) },
    });
    await openModelSelect();

    expect(screen.getByText('Create New Model')).toBeInTheDocument();
    expect(screen.getByText('Model A')).toBeInTheDocument();
  });

  test('shows the model-name input after choosing "Create New Model"', async () => {
    renderForm({
      ...minimalProps,
      modelByName: { 'Model A': mockModel('Model A', []) },
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
      modelByName: { 'Model A': mockModel('Model A', []) },
    });
    await openModelSelect();
    await userEvent.click(screen.getByRole('option', { name: 'Model A' }));

    await waitFor(async () => {
      expect((await innerRef.current.validateFields())[SELECTED_MODEL_FIELD]).toBe('Model A');
    });
    rerender(
      <RegisterModelForm {...minimalProps} innerRef={innerRef} modelByName={{ 'Model B': mockModel('Model B', []) }} />,
    );
    expect(screen.getByRole('textbox', { name: 'Model' })).toHaveValue('Model A');

    await openModelSelect();
    await userEvent.click(screen.getByRole('option', { name: 'Create New Model' }));
    expect(screen.getByLabelText('Model Name')).toBeInTheDocument();
    expect(innerRef.current.getFieldValue(SELECTED_MODEL_FIELD)).toBe(CREATE_NEW_MODEL_OPTION_VALUE);
  });

  test('searches registered models as the user types a model name', async () => {
    const onSearchRegisteredModels = jest.fn();
    renderForm({
      ...minimalProps,
      modelByName: { 'Model A': mockModel('Model A', []) },
      onSearchRegisteredModels,
    });
    await openModelSelect();
    await userEvent.click(screen.getByRole('textbox', { name: 'Model' }));
    await userEvent.paste('Model B');

    expect(onSearchRegisteredModels).toHaveBeenCalled();
  });
});
