import { render, screen } from '@testing-library/react';
import React from 'react';

import { IntlProvider } from '@databricks/i18n';

import { CategoricalAggregateChart } from './CategoricalAggregateChart';
import type { AssessmentInfo, AssessmentAggregates, AssessmentFilter } from '../../types';
import { getBarChartData } from '../../utils/AggregationUtils';
import { getDisplayScore } from '../../utils/DisplayUtils';

// Mock the utilities
jest.mock('../../utils/AggregationUtils', () => ({
  ERROR_KEY: 'Error',
  getBarChartData: jest.fn(),
}));

jest.mock('../../utils/DisplayUtils', () => ({
  getDisplayScore: jest.fn(),
  getDisplayScoreChange: jest.fn(),
}));

const mockGetBarChartData = jest.mocked(getBarChartData);
const mockGetDisplayScore = jest.mocked(getDisplayScore);

// Test wrapper with IntlProvider
const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en" messages={{}}>
    {children}
  </IntlProvider>
);

const mockTheme = {
  colors: {
    actionDefaultBackgroundHover: '#f0f0f0',
    actionDefaultBackgroundPress: '#e0e0e0',
    textSecondary: '#666',
    textValidationWarning: '#ff6b6b',
  },
  spacing: {
    xs: 8,
  },
  typography: {
    fontSizeSm: 12,
  },
  general: {
    borderRadiusBase: 4,
  },
} as any;

const mockIntl = {
  formatMessage: jest.fn((message, values) => {
    if (message.defaultMessage === '+{count} more') {
      return `+${values.count} more`;
    }
    return message.defaultMessage || '';
  }),
} as any;

const mockAssessmentInfo: AssessmentInfo = {
  name: 'test_assessment',
  displayName: 'Test Assessment',
  isKnown: true,
  isOverall: false,
  metricName: 'test_metric',
  isCustomMetric: false,
  isEditable: false,
  isRetrievalAssessment: false,
  dtype: 'string',
  uniqueValues: new Set(['value1', 'value2', 'value3']),
  docsLink: '',
  missingTooltip: '',
  description: '',
};

const mockAssessmentAggregates: AssessmentAggregates = {
  assessmentInfo: mockAssessmentInfo,
  currentCounts: new Map(),
  otherCounts: new Map(),
  currentNumRootCause: 0,
  otherNumRootCause: 0,
  assessmentFilters: [],
};

const mockAllAssessmentFilters: AssessmentFilter[] = [];

const mockToggleAssessmentFilter = jest.fn();

describe('CategoricalAggregateChart', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  const renderComponent = (props = {}) => {
    const defaultProps = {
      theme: mockTheme,
      intl: mockIntl,
      assessmentInfo: mockAssessmentInfo,
      assessmentAggregates: mockAssessmentAggregates,
      allAssessmentFilters: mockAllAssessmentFilters,
      toggleAssessmentFilter: mockToggleAssessmentFilter,
      currentRunDisplayName: 'Current Run',
      compareToRunDisplayName: 'Compare Run',
    };

    return render(
      <TestWrapper>
        <CategoricalAggregateChart {...defaultProps} {...props} />
      </TestWrapper>,
    );
  };

  describe('sorting behavior', () => {
    it('should maintain original order for pass-fail assessments', () => {
      const mockBarData = [
        {
          name: 'No',
          current: { value: 20, fraction: 0.8, isSelected: false, toggleFilter: jest.fn(), tooltip: 'No: 20' },
          backgroundColor: '#dc3545',
        },
        {
          name: 'Yes',
          current: { value: 5, fraction: 0.2, isSelected: false, toggleFilter: jest.fn(), tooltip: 'Yes: 5' },
          backgroundColor: '#28a745',
        },
      ];

      mockGetBarChartData.mockReturnValue(mockBarData);
      mockGetDisplayScore.mockReturnValue('80%');

      const passFailAssessmentInfo = { ...mockAssessmentInfo, dtype: 'pass-fail' as const };
      renderComponent({ assessmentInfo: passFailAssessmentInfo });

      const rows = screen.getAllByRole('row');
      expect(rows[0]).toHaveTextContent('No');
      expect(rows[1]).toHaveTextContent('Yes');
    });

    it('should maintain original order for boolean assessments', () => {
      const mockBarData = [
        {
          name: 'false',
          current: { value: 15, fraction: 0.75, isSelected: false, toggleFilter: jest.fn(), tooltip: 'false: 15' },
          backgroundColor: '#dc3545',
        },
        {
          name: 'true',
          current: { value: 5, fraction: 0.25, isSelected: false, toggleFilter: jest.fn(), tooltip: 'true: 5' },
          backgroundColor: '#28a745',
        },
      ];

      mockGetBarChartData.mockReturnValue(mockBarData);
      mockGetDisplayScore.mockReturnValue('75%');

      const booleanAssessmentInfo = { ...mockAssessmentInfo, dtype: 'boolean' as const };
      renderComponent({ assessmentInfo: booleanAssessmentInfo });

      const rows = screen.getAllByRole('row');
      expect(rows[0]).toHaveTextContent('false');
      expect(rows[1]).toHaveTextContent('true');
    });

    it('should sort by frequency for other assessment types', () => {
      const mockBarData = [
        {
          name: 'low',
          current: { value: 2, fraction: 0.1, isSelected: false, toggleFilter: jest.fn(), tooltip: 'low: 2' },
          backgroundColor: '#ffc107',
        },
        {
          name: 'high',
          current: { value: 15, fraction: 0.75, isSelected: false, toggleFilter: jest.fn(), tooltip: 'high: 15' },
          backgroundColor: '#28a745',
        },
        {
          name: 'medium',
          current: { value: 3, fraction: 0.15, isSelected: false, toggleFilter: jest.fn(), tooltip: 'medium: 3' },
          backgroundColor: '#fd7e14',
        },
      ];

      mockGetBarChartData.mockReturnValue(mockBarData);
      mockGetDisplayScore.mockReturnValue('75%');

      renderComponent();

      const rows = screen.getAllByRole('row');
      expect(rows[0]).toHaveTextContent('high');
      expect(rows[1]).toHaveTextContent('medium');
      expect(rows[2]).toHaveTextContent('low');
    });

    it('should put error and null entries at the bottom', () => {
      const mockBarData = [
        {
          name: 'Error',
          current: { value: 1, fraction: 0.05, isSelected: false, toggleFilter: jest.fn(), tooltip: 'Error: 1' },
          backgroundColor: '#dc3545',
        },
        {
          name: 'value1',
          current: { value: 10, fraction: 0.5, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value1: 10' },
          backgroundColor: '#007bff',
        },
        {
          name: 'null',
          current: { value: 2, fraction: 0.1, isSelected: false, toggleFilter: jest.fn(), tooltip: 'null: 2' },
          backgroundColor: '#6c757d',
        },
        {
          name: 'value2',
          current: { value: 7, fraction: 0.35, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value2: 7' },
          backgroundColor: '#28a745',
        },
      ];

      mockGetBarChartData.mockReturnValue(mockBarData);
      mockGetDisplayScore.mockReturnValue('50%');

      renderComponent();

      const rows = screen.getAllByRole('row');
      expect(rows[0]).toHaveTextContent('value1');
      expect(rows[1]).toHaveTextContent('value2');
      expect(rows[2]).toHaveTextContent('Error');
      expect(rows[3]).toHaveTextContent('null');
    });
  });

  describe('popover behavior', () => {
    it('should show popover when there are more than 4 items', () => {
      const mockBarData = [
        {
          name: 'value1',
          current: { value: 10, fraction: 0.4, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value1: 10' },
          backgroundColor: '#007bff',
        },
        {
          name: 'value2',
          current: { value: 8, fraction: 0.32, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value2: 8' },
          backgroundColor: '#28a745',
        },
        {
          name: 'value3',
          current: { value: 4, fraction: 0.16, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value3: 4' },
          backgroundColor: '#ffc107',
        },
        {
          name: 'value4',
          current: { value: 2, fraction: 0.08, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value4: 2' },
          backgroundColor: '#fd7e14',
        },
        {
          name: 'value5',
          current: { value: 1, fraction: 0.04, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value5: 1' },
          backgroundColor: '#6c757d',
        },
      ];

      mockGetBarChartData.mockReturnValue(mockBarData);
      mockGetDisplayScore.mockReturnValue('40%');

      renderComponent();

      expect(screen.getByText('value1')).toBeInTheDocument();
      expect(screen.getByText('value2')).toBeInTheDocument();
      expect(screen.getByText('value3')).toBeInTheDocument();
      expect(screen.queryByText('value4')).not.toBeInTheDocument();
      expect(screen.queryByText('value5')).not.toBeInTheDocument();
      expect(screen.getByText('+2 more')).toBeInTheDocument();
    });

    it('should not show popover when there are 4 or fewer items', () => {
      const mockBarData = [
        {
          name: 'value1',
          current: { value: 10, fraction: 0.4, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value1: 10' },
          backgroundColor: '#007bff',
        },
        {
          name: 'value2',
          current: { value: 8, fraction: 0.32, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value2: 8' },
          backgroundColor: '#28a745',
        },
        {
          name: 'value3',
          current: { value: 4, fraction: 0.16, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value3: 4' },
          backgroundColor: '#ffc107',
        },
        {
          name: 'value4',
          current: { value: 2, fraction: 0.08, isSelected: false, toggleFilter: jest.fn(), tooltip: 'value4: 2' },
          backgroundColor: '#fd7e14',
        },
      ];

      mockGetBarChartData.mockReturnValue(mockBarData);
      mockGetDisplayScore.mockReturnValue('40%');

      renderComponent();

      expect(screen.getByText('value1')).toBeInTheDocument();
      expect(screen.getByText('value2')).toBeInTheDocument();
      expect(screen.getByText('value3')).toBeInTheDocument();
      expect(screen.getByText('value4')).toBeInTheDocument();
      expect(screen.queryByText('+0 more')).not.toBeInTheDocument();
    });
  });
});
