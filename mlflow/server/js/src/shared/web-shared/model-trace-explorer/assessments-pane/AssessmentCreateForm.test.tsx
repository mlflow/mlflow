import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { AssessmentCreateForm } from './AssessmentCreateForm';
import type { Assessment } from '../ModelTrace.types';
import { MOCK_ASSESSMENT, MOCK_EXPECTATION } from '../ModelTraceExplorer.test-utils';
import { AssessmentSchemaContextProvider } from '../contexts/AssessmentSchemaContext';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

// Mock the hooks
jest.mock('../hooks/useCreateAssessment', () => ({
  useCreateAssessment: jest.fn(() => ({
    createAssessmentMutation: jest.fn(),
    isLoading: false,
  })),
}));

const TestWrapper = ({ children, assessments = [] }: { children: React.ReactNode; assessments?: Assessment[] }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <AssessmentSchemaContextProvider assessments={assessments}>{children}</AssessmentSchemaContextProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>
  );
};

describe('AssessmentCreateForm', () => {
  const mockSetExpanded = jest.fn();
  const defaultProps = {
    traceId: 'test-trace-id',
    assessmentType: 'feedback' as const,
    setExpanded: mockSetExpanded,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('handleChangeSchema - existing schemas', () => {
    it('should update data type when selecting an existing schema', async () => {
      const user = userEvent.setup();

      // Mock existing assessments to populate schemas
      const existingAssessments: Assessment[] = [
        MOCK_ASSESSMENT, // name: 'Relevance', feedback (string type)
        MOCK_EXPECTATION, // name: 'expected_facts', expectation (json type)
      ];

      render(
        <TestWrapper assessments={existingAssessments}>
          <AssessmentCreateForm {...defaultProps} assessmentType="expectation" />
        </TestWrapper>,
      );

      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;
      expect(dataTypeSelect).toBeInTheDocument();
      expect(dataTypeSelect).toHaveTextContent('Boolean');

      // Change data type to number
      await user.click(dataTypeSelect);
      await user.click(screen.getAllByText('Number')[0]);
      await waitFor(() => {
        expect(dataTypeSelect).toHaveTextContent('Number');
      });

      // Open the name typeahead
      const nameInput = screen.getByPlaceholderText('Enter an expectation name');
      await user.click(nameInput);

      // Select an existing schema (expected_facts which is expectation/json)
      await waitFor(() => {
        expect(screen.getByText('expected_facts')).toBeInTheDocument();
      });
      await user.click(screen.getByText('expected_facts'));

      // Data type should be updated to match the schema
      await waitFor(() => {
        expect(nameInput).toHaveValue('expected_facts');
        expect(dataTypeSelect).toHaveTextContent('JSON');
      });
    });
  });

  describe('handleChangeSchema - data type clamping', () => {
    it('should clamp JSON data type to string when selecting an existing schema in feedback form', async () => {
      const user = userEvent.setup();

      // MOCK_EXPECTATION has dataType 'json'
      const existingAssessments: Assessment[] = [MOCK_EXPECTATION];

      render(
        <TestWrapper assessments={existingAssessments}>
          <AssessmentCreateForm {...defaultProps} assessmentType="feedback" />
        </TestWrapper>,
      );

      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      // Open the name typeahead and select the expectation schema (json type)
      const nameInput = screen.getByPlaceholderText('Enter a feedback name');
      await user.click(nameInput);

      await waitFor(() => {
        expect(screen.getByText('expected_facts')).toBeInTheDocument();
      });
      await user.click(screen.getByText('expected_facts'));

      // Data type should be clamped to string (not json) since this is a feedback form
      await waitFor(() => {
        expect(nameInput).toHaveValue('expected_facts');
        expect(dataTypeSelect).toHaveTextContent('String');
      });
    });
  });

  describe('handleChangeSchema - new assessment names', () => {
    it('should preserve user-selected data type when typing a new assessment name', async () => {
      const user = userEvent.setup();

      // Mock existing assessments to populate schemas
      const existingAssessments: Assessment[] = [MOCK_ASSESSMENT];

      render(
        <TestWrapper assessments={existingAssessments}>
          <AssessmentCreateForm {...defaultProps} />
        </TestWrapper>,
      );

      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      // Change data type to number
      await user.click(dataTypeSelect);
      await user.click(screen.getByText('Number'));
      await waitFor(() => {
        expect(dataTypeSelect).toHaveTextContent('Number');
      });

      // Type a new assessment name that doesn't exist in schemas
      const nameInput = screen.getByPlaceholderText('Enter a feedback name');
      await user.type(nameInput, 'my_new_assessment');

      // Open the dropdown
      await user.click(nameInput);

      // The new name should appear in the typeahead
      await waitFor(() => {
        expect(screen.getByText('my_new_assessment')).toBeInTheDocument();
      });

      // Select the new name
      await user.click(screen.getByText('my_new_assessment'));

      // Name should be updated, but data type should be preserved
      await waitFor(() => {
        expect(nameInput).toHaveValue('my_new_assessment');
        expect(dataTypeSelect).toHaveTextContent('Number');
      });
    });

    it('should preserve user-selected data type when pressing Enter on a new name', async () => {
      const user = userEvent.setup();

      const existingAssessments: Assessment[] = [MOCK_ASSESSMENT];

      render(
        <TestWrapper assessments={existingAssessments}>
          <AssessmentCreateForm {...defaultProps} />
        </TestWrapper>,
      );

      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      // Change to non-default value
      await user.click(dataTypeSelect);
      await user.click(screen.getAllByText('String')[0]);

      await waitFor(() => {
        expect(dataTypeSelect).toHaveTextContent('String');
      });

      // Type a new name and press Enter
      const nameInput = screen.getByPlaceholderText('Enter a feedback name');
      await user.type(nameInput, 'another_new_name');
      await user.keyboard('{Enter}');

      // Data type should be preserved
      await waitFor(() => {
        expect(nameInput).toHaveValue('another_new_name');
        expect(dataTypeSelect).toHaveTextContent('String');
      });
    });
  });

  describe('handleChangeSchema - clearing selection', () => {
    it('should reset fields when clearing the name', async () => {
      const user = userEvent.setup();

      render(
        <TestWrapper assessments={[MOCK_ASSESSMENT]}>
          <AssessmentCreateForm {...defaultProps} />
        </TestWrapper>,
      );

      const nameInput = screen.getByPlaceholderText('Enter a feedback name');
      await user.type(nameInput, 'test_name');

      // Clear the input (simulating the clear button)
      await user.clear(nameInput);

      // Fields should reset to defaults
      await waitFor(() => {
        expect(nameInput).toHaveValue('');
      });
    });
  });

  describe('edge cases', () => {
    it('should handle empty schemas list', async () => {
      const user = userEvent.setup();

      render(
        <TestWrapper assessments={[]}>
          <AssessmentCreateForm {...defaultProps} />
        </TestWrapper>,
      );

      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      // Change to non-default value
      await user.click(dataTypeSelect);
      await user.click(screen.getByText('Number'));

      await waitFor(() => {
        expect(dataTypeSelect).toHaveTextContent('Number');
      });

      // Type a name when there are no existing schemas
      const nameInput = screen.getByPlaceholderText('Enter a feedback name');
      await user.type(nameInput, 'first_assessment');
      await user.keyboard('{Enter}');

      // Data type should be preserved since this is a new name
      await waitFor(() => {
        expect(nameInput).toHaveValue('first_assessment');
        expect(dataTypeSelect).toHaveTextContent('Number');
      });
    });

    it('should handle schema with same name as typed value', async () => {
      const user = userEvent.setup();

      const existingAssessments: Assessment[] = [MOCK_ASSESSMENT]; // name: 'Relevance'

      render(
        <TestWrapper assessments={existingAssessments}>
          <AssessmentCreateForm {...defaultProps} />
        </TestWrapper>,
      );

      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      // Type the exact name of an existing schema
      const nameInput = screen.getByPlaceholderText('Enter a feedback name');
      await user.type(nameInput, 'Relevance');
      await user.click(nameInput);

      // Select it from the dropdown
      await waitFor(() => {
        expect(screen.getByTestId('assessment-name-typeahead-item-Relevance')).toBeInTheDocument();
      });
      await user.click(screen.getByTestId('assessment-name-typeahead-item-Relevance'));

      // Since 'Relevance' is an existing schema (feedback/string), it should update data type
      await waitFor(() => {
        expect(nameInput).toHaveValue('Relevance');
        expect(dataTypeSelect).toHaveTextContent('String');
      });
    });
  });
});
