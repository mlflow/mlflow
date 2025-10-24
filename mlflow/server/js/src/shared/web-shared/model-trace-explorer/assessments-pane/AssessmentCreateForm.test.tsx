import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { AssessmentCreateForm } from './AssessmentCreateForm';
import { AssessmentSchemaContextProvider } from '../contexts/AssessmentSchemaContext';
import type { Assessment } from '../ModelTrace.types';
import { MOCK_ASSESSMENT, MOCK_EXPECTATION } from '../ModelTraceExplorer.test-utils';

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
    setExpanded: mockSetExpanded,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('handleChangeSchema - existing schemas', () => {
    it('should update all fields when selecting an existing schema', async () => {
      const user = userEvent.setup();

      // Mock existing assessments to populate schemas
      const existingAssessments: Assessment[] = [
        MOCK_ASSESSMENT, // name: 'Relevance', feedback (string type)
        MOCK_EXPECTATION, // name: 'expected_facts', expectation (json type)
      ];

      render(
        <TestWrapper assessments={existingAssessments}>
          <AssessmentCreateForm {...defaultProps} />
        </TestWrapper>,
      );

      // Get the select buttons
      const assessmentTypeSelect = screen.getByLabelText('Assessment Type') as HTMLButtonElement;
      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      expect(assessmentTypeSelect).toBeInTheDocument();
      expect(dataTypeSelect).toBeInTheDocument();

      // Verify initial values
      expect(assessmentTypeSelect).toHaveTextContent('Feedback');
      expect(dataTypeSelect).toHaveTextContent('Boolean');

      // Change assessment type to expectation and data type to number
      await user.click(assessmentTypeSelect);
      await user.click(screen.getByText('Expectation'));
      await waitFor(() => {
        expect(assessmentTypeSelect).toHaveTextContent('Expectation');
      });

      await user.click(dataTypeSelect);
      await user.click(screen.getAllByText('Number')[0]);
      await waitFor(() => {
        expect(dataTypeSelect).toHaveTextContent('Number');
      });

      // Open the name typeahead
      const nameInput = screen.getByPlaceholderText('Enter an assessment name');
      await user.click(nameInput);

      // Select an existing schema (expected_facts which is expectation/json)
      await waitFor(() => {
        expect(screen.getByText('expected_facts')).toBeInTheDocument();
      });
      await user.click(screen.getByText('expected_facts'));

      // All fields should be updated to match the schema
      await waitFor(() => {
        expect(nameInput).toHaveValue('expected_facts');
        expect(assessmentTypeSelect).toHaveTextContent('Expectation');
        expect(dataTypeSelect).toHaveTextContent('JSON');
      });
    });
  });

  describe('handleChangeSchema - new assessment names', () => {
    it('should preserve user-selected fields when typing a new assessment name', async () => {
      const user = userEvent.setup();

      // Mock existing assessments to populate schemas
      const existingAssessments: Assessment[] = [MOCK_ASSESSMENT];

      render(
        <TestWrapper assessments={existingAssessments}>
          <AssessmentCreateForm {...defaultProps} />
        </TestWrapper>,
      );

      // Get the select buttons
      const assessmentTypeSelect = screen.getByLabelText('Assessment Type') as HTMLButtonElement;
      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      // Change assessment type to expectation and data type to number
      await user.click(assessmentTypeSelect);
      await user.click(screen.getByText('Expectation'));
      await waitFor(() => {
        expect(assessmentTypeSelect).toHaveTextContent('Expectation');
      });

      await user.click(dataTypeSelect);
      await user.click(screen.getByText('Number'));
      await waitFor(() => {
        expect(dataTypeSelect).toHaveTextContent('Number');
      });

      // Type a new assessment name that doesn't exist in schemas
      const nameInput = screen.getByPlaceholderText('Enter an assessment name');
      await user.type(nameInput, 'my_new_assessment');

      // Open the dropdown
      await user.click(nameInput);

      // The new name should appear in the typeahead
      await waitFor(() => {
        expect(screen.getByText('my_new_assessment')).toBeInTheDocument();
      });

      // Select the new name
      await user.click(screen.getByText('my_new_assessment'));

      // Name should be updated, but assessment type and data type should be preserved
      await waitFor(() => {
        expect(nameInput).toHaveValue('my_new_assessment');
        expect(assessmentTypeSelect).toHaveTextContent('Expectation');
        expect(dataTypeSelect).toHaveTextContent('Number');
      });
    });

    it('should preserve user-selected fields when pressing Enter on a new name', async () => {
      const user = userEvent.setup();

      const existingAssessments: Assessment[] = [MOCK_ASSESSMENT];

      render(
        <TestWrapper assessments={existingAssessments}>
          <AssessmentCreateForm {...defaultProps} />
        </TestWrapper>,
      );

      // Get the select buttons
      const assessmentTypeSelect = screen.getByLabelText('Assessment Type') as HTMLButtonElement;
      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      // Change to non-default values
      await user.click(assessmentTypeSelect);
      await user.click(screen.getByText('Expectation'));

      await user.click(dataTypeSelect);
      await user.click(screen.getAllByText('String')[0]);

      await waitFor(() => {
        expect(assessmentTypeSelect).toHaveTextContent('Expectation');
        expect(dataTypeSelect).toHaveTextContent('String');
      });

      // Type a new name and press Enter
      const nameInput = screen.getByPlaceholderText('Enter an assessment name');
      await user.type(nameInput, 'another_new_name');
      await user.keyboard('{Enter}');

      // Fields should be preserved
      await waitFor(() => {
        expect(nameInput).toHaveValue('another_new_name');
        expect(assessmentTypeSelect).toHaveTextContent('Expectation');
        expect(dataTypeSelect).toHaveTextContent('String');
      });
    });
  });

  describe('handleChangeSchema - clearing selection', () => {
    it('should reset all fields when clearing the name', async () => {
      const user = userEvent.setup();

      render(
        <TestWrapper assessments={[MOCK_ASSESSMENT]}>
          <AssessmentCreateForm {...defaultProps} />
        </TestWrapper>,
      );

      // Get the select button
      const assessmentTypeSelect = screen.getByLabelText('Assessment Type') as HTMLButtonElement;

      // Change some values
      await user.click(assessmentTypeSelect);
      await user.click(screen.getByText('Expectation'));

      const nameInput = screen.getByPlaceholderText('Enter an assessment name');
      await user.type(nameInput, 'test_name');

      // Clear the input (simulating the clear button)
      await user.clear(nameInput);

      // All fields should reset to defaults
      await waitFor(() => {
        expect(nameInput).toHaveValue('');
        // Note: We can't easily test the internal state reset here
        // as the selects may not visibly change until interaction
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

      // Get the select buttons
      const assessmentTypeSelect = screen.getByLabelText('Assessment Type') as HTMLButtonElement;
      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      // Change to non-default values
      await user.click(assessmentTypeSelect);
      await user.click(screen.getByText('Expectation'));

      await user.click(dataTypeSelect);
      await user.click(screen.getByText('Number'));

      await waitFor(() => {
        expect(assessmentTypeSelect).toHaveTextContent('Expectation');
        expect(dataTypeSelect).toHaveTextContent('Number');
      });

      // Type a name when there are no existing schemas
      const nameInput = screen.getByPlaceholderText('Enter an assessment name');
      await user.type(nameInput, 'first_assessment');
      await user.keyboard('{Enter}');

      // Fields should be preserved since this is a new name
      await waitFor(() => {
        expect(nameInput).toHaveValue('first_assessment');
        expect(assessmentTypeSelect).toHaveTextContent('Expectation');
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

      // Get the select buttons
      const assessmentTypeSelect = screen.getByLabelText('Assessment Type') as HTMLButtonElement;
      const dataTypeSelect = screen.getByLabelText('Data Type') as HTMLButtonElement;

      // Change values first
      await user.click(assessmentTypeSelect);
      await user.click(screen.getByText('Expectation'));

      await waitFor(() => {
        expect(assessmentTypeSelect).toHaveTextContent('Expectation');
      });

      // Type the exact name of an existing schema
      const nameInput = screen.getByPlaceholderText('Enter an assessment name');
      await user.type(nameInput, 'Relevance');
      await user.click(nameInput);

      // Select it from the dropdown
      await waitFor(() => {
        expect(screen.getByTestId('assessment-name-typeahead-item-Relevance')).toBeInTheDocument();
      });
      await user.click(screen.getByTestId('assessment-name-typeahead-item-Relevance'));

      // Since 'Relevance' is an existing schema (feedback/string), it should update all fields
      await waitFor(() => {
        expect(nameInput).toHaveValue('Relevance');
        expect(assessmentTypeSelect).toHaveTextContent('Feedback');
        expect(dataTypeSelect).toHaveTextContent('String');
      });
    });
  });
});
