import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import React, { Component } from 'react';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem } from '../../../common/utils/TestUtils.react18';
import { GenericInputModal } from './GenericInputModal';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';

// Minimal stand-in for the antd form that GenericInputModal drives through innerRef
class FakeForm extends Component<{ innerRef?: any; fieldValidation?: () => Promise<any> }> {
  componentDidMount() {
    this.props.innerRef.current = {
      validateFields: this.props.fieldValidation ?? (() => Promise.resolve({ name: 'value' })),
      resetFields: jest.fn(),
    };
  }

  render() {
    return <div data-testid="fake-form" />;
  }
}

type RenderProps = Partial<React.ComponentProps<typeof GenericInputModal>> & {
  fieldValidation?: () => Promise<any>;
};

const renderModal = ({ fieldValidation, ...props }: RenderProps = {}) =>
  renderWithDesignSystem(
    <GenericInputModal isOpen title="Create Experiment" okText="Create" onClose={jest.fn()} {...(props as any)}>
      <FakeForm fieldValidation={fieldValidation} />
    </GenericInputModal>,
  );

describe('GenericInputModal inline submission errors', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('shows the server-provided message inline when submission fails with an ErrorWrapper', async () => {
    const handleSubmit = jest
      .fn<() => Promise<never>>()
      .mockRejectedValue(
        new ErrorWrapper(
          JSON.stringify({ error_code: 'INVALID_PARAMETER_VALUE', message: 'Invalid artifact location' }),
          400,
        ),
      );
    renderModal({ handleSubmit });

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(screen.getByText('Invalid artifact location')).toBeInTheDocument();
    });
    expect(handleSubmit).toHaveBeenCalled();
  });

  test('shows plain Error messages inline', async () => {
    const handleSubmit = jest.fn<() => Promise<never>>().mockRejectedValue(new Error('Network unreachable'));
    renderModal({ handleSubmit });

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(screen.getByText('Network unreachable')).toBeInTheDocument();
    });
  });

  test('shows a fallback message when the error carries no details', async () => {
    const handleSubmit = jest.fn<() => Promise<never>>().mockRejectedValue(new Error());
    renderModal({ handleSubmit });

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(screen.getByText('The request failed. Please try again.')).toBeInTheDocument();
    });
  });

  test('does not show a modal-level error for form validation failures', async () => {
    const handleSubmit = jest.fn<() => Promise<any>>();
    renderModal({
      handleSubmit,
      fieldValidation: () => Promise.reject({ errorFields: [{ name: ['name'], errors: ['Name is required'] }] }),
    });

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(handleSubmit).not.toHaveBeenCalled();
    });
    expect(document.querySelector('[role="alert"]')).not.toBeInTheDocument();
  });

  test('clears the inline error on the next submission attempt', async () => {
    const handleSubmit = jest
      .fn<() => Promise<any>>()
      .mockRejectedValueOnce(new Error('Transient failure'))
      .mockResolvedValueOnce(undefined);
    const onClose = jest.fn();
    renderModal({ handleSubmit, onClose });

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => {
      expect(screen.getByText('Transient failure')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => {
      expect(screen.queryByText('Transient failure')).not.toBeInTheDocument();
    });
    expect(onClose).toHaveBeenCalled();
  });
});
