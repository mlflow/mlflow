import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';

import { IntlProvider } from '@databricks/i18n';

import { EvaluationsReviewAssessmentUpsertForm } from './EvaluationsReviewAssessmentUpsertForm';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(50000); // Larger timeout for heavier UI test

describe('EvaluationsReviewAssessmentUpsertForm', () => {
  const onSaveMock = jest.fn();
  const onCancelMock = jest.fn();

  const renderTestComponent = (
    overrideProps: Partial<ComponentProps<typeof EvaluationsReviewAssessmentUpsertForm>> = {},
  ) =>
    render(
      <EvaluationsReviewAssessmentUpsertForm
        onCancel={onCancelMock}
        onSave={onSaveMock}
        valueSuggestions={[]}
        {...overrideProps}
      />,
      { wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider> },
    );

  beforeEach(() => {
    onSaveMock.mockClear();
    onCancelMock.mockClear();

    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    const originalConsoleError = console.error;

    // Silence a noisy issue with Typeahead component and its '_TYPE' prop
    jest.spyOn(console, 'error').mockImplementation((message, ...args) => {
      if (message.includes('React does not recognize the `%s` prop on a DOM element')) {
        return;
      }
      originalConsoleError(message, ...args);
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('will render and display suggestions for known assessment', async () => {
    renderTestComponent({
      valueSuggestions: [
        { label: 'Pass', key: 'yes' },
        { label: 'Fail', key: 'no' },
      ],
      editedAssessment: {
        name: 'overall_assessment',
      } as any,
    });

    await userEvent.click(screen.getByPlaceholderText('Select or type an assessment'));
    await userEvent.click(screen.getByText('Pass'));
    await userEvent.click(screen.getByText('Confirm'));

    expect(onSaveMock).toHaveBeenLastCalledWith({
      assessmentName: 'overall_assessment',
      rationale: undefined,
      value: 'yes',
    });
  });

  test('will render and display suggestions for all assessments', async () => {
    renderTestComponent({
      valueSuggestions: [
        { label: 'Relevant', key: 'yes', rootAssessmentName: 'relevancy' },
        { label: 'Irrelevant', key: 'no', rootAssessmentName: 'relevancy' },
        { label: 'Correct', key: 'yes', rootAssessmentName: 'correctness' },
        { label: 'Incorrect', key: 'no', rootAssessmentName: 'correctness' },
      ],
    });

    const inputElement = screen.getByPlaceholderText('Select or type an assessment');

    await userEvent.click(inputElement);
    await userEvent.click(screen.getByText('Relevant'));
    await userEvent.click(screen.getByText('Confirm'));

    expect(onSaveMock).toHaveBeenLastCalledWith({
      assessmentName: 'relevancy',
      rationale: undefined,
      value: 'yes',
    });
  });

  test('will allow entering custom suggestion', async () => {
    renderTestComponent({
      valueSuggestions: [
        { label: 'Relevant', key: 'yes', rootAssessmentName: 'relevancy' },
        { label: 'Irrelevant', key: 'no', rootAssessmentName: 'relevancy' },
        { label: 'Correct', key: 'yes', rootAssessmentName: 'correctness' },
        { label: 'Incorrect', key: 'no', rootAssessmentName: 'correctness' },
      ],
    });

    const inputElement = screen.getByPlaceholderText('Select or type an assessment');
    const rationaleInputElement = screen.getByPlaceholderText('Add rationale (optional)');

    await userEvent.click(inputElement);
    await userEvent.clear(inputElement);
    await userEvent.type(inputElement, 'incorrect');

    expect(screen.getByText('Incorrect')).toBeInTheDocument();
    expect(screen.queryByText('Correct')).not.toBeInTheDocument();

    await userEvent.clear(inputElement);
    await userEvent.click(inputElement);
    await userEvent.paste('some_custom_value');

    expect(screen.getByText('Add "some_custom_value"')).toBeInTheDocument();

    await userEvent.click(rationaleInputElement);
    await userEvent.paste('test rationale');

    await userEvent.click(screen.getByText('Confirm'));

    await waitFor(() => {
      expect(onSaveMock).toHaveBeenLastCalledWith({
        assessmentName: 'some_custom_value',
        value: true,
        rationale: 'test rationale',
      });
    });
  });

  test('will disallow entering custom suggestion on readonly fields', async () => {
    renderTestComponent({
      editedAssessment: {
        stringValue: 'yes',
        name: 'relevancy',
      } as any,
      valueSuggestions: [
        { label: 'Relevant', key: 'yes', rootAssessmentName: 'relevancy' },
        { label: 'Irrelevant', key: 'no', rootAssessmentName: 'relevancy' },
      ],
      readOnly: true,
    });

    const inputElement = screen.getByPlaceholderText('Select or type an assessment');

    expect(inputElement).toHaveAttribute('readonly');

    // Attempt to clear the input should throw an error
    await expect(() => userEvent.clear(inputElement)).rejects.toThrow();
  });

  test('will display and repeat existing rationale', async () => {
    renderTestComponent({
      valueSuggestions: [
        { label: 'Pass', key: 'yes' },
        { label: 'Fail', key: 'no' },
      ],
      editedAssessment: {
        name: 'overall_assessment',
        rationale: 'test rationale',
        stringValue: 'yes',
      } as any,
    });

    expect(screen.getByPlaceholderText('Add rationale (optional)')).toHaveValue('test rationale');

    await userEvent.click(screen.getByText('Confirm'));

    expect(onSaveMock).toHaveBeenLastCalledWith({
      assessmentName: 'overall_assessment',
      rationale: 'test rationale',
      value: 'yes',
    });
  });
});
