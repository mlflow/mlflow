import { describe, it, expect } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { AssessmentSummaryTable } from './AssessmentSummaryTable';
import { DesignSystemProvider } from '@databricks/design-system';

describe('AssessmentSummaryTable', () => {
  const renderComponent = (props: {
    assessmentNames: string[];
    countsByName: Map<string, number>;
    avgValuesByName: Map<string, number>;
  }) => {
    return renderWithIntl(
      <DesignSystemProvider>
        <AssessmentSummaryTable {...props} />
      </DesignSystemProvider>,
    );
  };

  describe('basic rendering', () => {
    it('should display the section title', () => {
      renderComponent({
        assessmentNames: ['accuracy'],
        countsByName: new Map([['accuracy', 100]]),
        avgValuesByName: new Map([['accuracy', 0.85]]),
      });

      expect(screen.getByText('Quality Summary')).toBeInTheDocument();
    });

    it('should display table column headers', () => {
      renderComponent({
        assessmentNames: ['accuracy'],
        countsByName: new Map([['accuracy', 100]]),
        avgValuesByName: new Map([['accuracy', 0.85]]),
      });

      expect(screen.getByText('Scorer')).toBeInTheDocument();
      expect(screen.getByText('Total Count')).toBeInTheDocument();
      expect(screen.getByText('Average Value')).toBeInTheDocument();
    });

    it('should display scorer names', () => {
      renderComponent({
        assessmentNames: ['accuracy', 'relevance', 'coherence'],
        countsByName: new Map([
          ['accuracy', 100],
          ['relevance', 200],
          ['coherence', 150],
        ]),
        avgValuesByName: new Map([
          ['accuracy', 0.85],
          ['relevance', 0.92],
          ['coherence', 0.78],
        ]),
      });

      expect(screen.getByText('accuracy')).toBeInTheDocument();
      expect(screen.getByText('relevance')).toBeInTheDocument();
      expect(screen.getByText('coherence')).toBeInTheDocument();
    });

    it('should display formatted counts', () => {
      renderComponent({
        assessmentNames: ['high_count', 'low_count'],
        countsByName: new Map([
          ['high_count', 15000],
          ['low_count', 500],
        ]),
        avgValuesByName: new Map([
          ['high_count', 0.9],
          ['low_count', 0.8],
        ]),
      });

      // 15000 should be formatted as 15.00K
      expect(screen.getByText('15.00K')).toBeInTheDocument();
      // 500 should remain as is
      expect(screen.getByText('500')).toBeInTheDocument();
    });

    it('should display average values with 2 decimal places', () => {
      renderComponent({
        assessmentNames: ['scorer1'],
        countsByName: new Map([['scorer1', 100]]),
        avgValuesByName: new Map([['scorer1', 0.8567]]),
      });

      expect(screen.getByText('0.86')).toBeInTheDocument();
    });

    it('should display dash for non-numeric assessments', () => {
      renderComponent({
        assessmentNames: ['numeric_scorer', 'text_scorer'],
        countsByName: new Map([
          ['numeric_scorer', 100],
          ['text_scorer', 50],
        ]),
        avgValuesByName: new Map([['numeric_scorer', 0.9]]), // text_scorer has no avg
      });

      expect(screen.getByText('0.90')).toBeInTheDocument();
      expect(screen.getByText('-')).toBeInTheDocument();
    });
  });

  describe('sorting functionality', () => {
    const defaultProps = {
      assessmentNames: ['alpha_scorer', 'beta_scorer', 'gamma_scorer'],
      countsByName: new Map([
        ['alpha_scorer', 300],
        ['beta_scorer', 100],
        ['gamma_scorer', 200],
      ]),
      avgValuesByName: new Map([
        ['alpha_scorer', 0.7],
        ['beta_scorer', 0.9],
        ['gamma_scorer', 0.5],
      ]),
    };

    it('should sort by total count descending by default', () => {
      renderComponent(defaultProps);

      const scorerNames = screen.getAllByText(/alpha_scorer|beta_scorer|gamma_scorer/);
      expect(scorerNames[0].textContent).toBe('alpha_scorer'); // 300 counts
      expect(scorerNames[1].textContent).toBe('gamma_scorer'); // 200 counts
      expect(scorerNames[2].textContent).toBe('beta_scorer'); // 100 counts
    });

    it('should toggle sort direction when clicking the same column', async () => {
      renderComponent(defaultProps);

      // Click Total Count header to toggle to ascending
      const countHeader = screen.getByRole('button', { name: /Total Count/i });
      await userEvent.click(countHeader);

      // Now should be ascending - beta_scorer first (100 counts)
      const scorerNames = screen.getAllByText(/alpha_scorer|beta_scorer|gamma_scorer/);
      expect(scorerNames[0].textContent).toBe('beta_scorer');
      expect(scorerNames[1].textContent).toBe('gamma_scorer');
      expect(scorerNames[2].textContent).toBe('alpha_scorer');
    });

    it('should sort by scorer name when clicking Scorer header', async () => {
      renderComponent(defaultProps);

      // Click Scorer header
      const scorerHeader = screen.getByRole('button', { name: /^Scorer$/i });
      await userEvent.click(scorerHeader);

      // Should sort by name descending first (gamma > beta > alpha)
      const scorerNames = screen.getAllByText(/alpha_scorer|beta_scorer|gamma_scorer/);
      expect(scorerNames[0].textContent).toBe('gamma_scorer');
      expect(scorerNames[1].textContent).toBe('beta_scorer');
      expect(scorerNames[2].textContent).toBe('alpha_scorer');
    });

    it('should sort by average value when clicking Average Value header', async () => {
      renderComponent(defaultProps);

      // Click Average Value header
      const avgHeader = screen.getByRole('button', { name: /Average Value/i });
      await userEvent.click(avgHeader);

      // Should sort by avg value descending
      // beta_scorer: 0.9, alpha_scorer: 0.7, gamma_scorer: 0.5
      const scorerNames = screen.getAllByText(/alpha_scorer|beta_scorer|gamma_scorer/);
      expect(scorerNames[0].textContent).toBe('beta_scorer');
      expect(scorerNames[1].textContent).toBe('alpha_scorer');
      expect(scorerNames[2].textContent).toBe('gamma_scorer');
    });

    it('should handle sorting with undefined average values', async () => {
      renderComponent({
        assessmentNames: ['numeric_a', 'text_b', 'numeric_c'],
        countsByName: new Map([
          ['numeric_a', 100],
          ['text_b', 100],
          ['numeric_c', 100],
        ]),
        avgValuesByName: new Map([
          ['numeric_a', 0.5],
          // text_b has no avg value
          ['numeric_c', 0.8],
        ]),
      });

      // Click Average Value header
      const avgHeader = screen.getByRole('button', { name: /Average Value/i });
      await userEvent.click(avgHeader);

      // Should sort by avg value descending, undefined treated as lowest
      const scorerNames = screen.getAllByText(/numeric_a|text_b|numeric_c/);
      expect(scorerNames[0].textContent).toBe('numeric_c'); // 0.8
      expect(scorerNames[1].textContent).toBe('numeric_a'); // 0.5
      expect(scorerNames[2].textContent).toBe('text_b'); // undefined (treated as -Infinity)
    });

    it('should support keyboard navigation for sorting', async () => {
      renderComponent(defaultProps);

      // Focus and press Enter on Scorer header
      const scorerHeader = screen.getByRole('button', { name: /^Scorer$/i });
      scorerHeader.focus();
      await userEvent.keyboard('{Enter}');

      // Should sort by name descending
      const scorerNames = screen.getAllByText(/alpha_scorer|beta_scorer|gamma_scorer/);
      expect(scorerNames[0].textContent).toBe('gamma_scorer');
      expect(scorerNames[1].textContent).toBe('beta_scorer');
      expect(scorerNames[2].textContent).toBe('alpha_scorer');
    });

    it('should display sort icon on active column', () => {
      renderComponent(defaultProps);

      // Default is Total Count descending - check for sort icon
      const countHeader = screen.getByRole('button', { name: /Total Count/i });
      expect(within(countHeader).getByRole('img', { hidden: true })).toBeInTheDocument();

      // Scorer header should not have sort icon
      const scorerHeader = screen.getByRole('button', { name: /^Scorer$/i });
      expect(within(scorerHeader).queryByRole('img', { hidden: true })).not.toBeInTheDocument();
    });
  });

  describe('edge cases', () => {
    it('should handle empty data', () => {
      renderComponent({
        assessmentNames: [],
        countsByName: new Map(),
        avgValuesByName: new Map(),
      });

      // Should still render headers
      expect(screen.getByText('Quality Summary')).toBeInTheDocument();
      expect(screen.getByText('Scorer')).toBeInTheDocument();
    });

    it('should handle single assessment', () => {
      renderComponent({
        assessmentNames: ['single_scorer'],
        countsByName: new Map([['single_scorer', 42]]),
        avgValuesByName: new Map([['single_scorer', 1.0]]),
      });

      expect(screen.getByText('single_scorer')).toBeInTheDocument();
      expect(screen.getByText('42')).toBeInTheDocument();
      expect(screen.getByText('1.00')).toBeInTheDocument();
    });

    it('should handle zero values', () => {
      renderComponent({
        assessmentNames: ['zero_scorer'],
        countsByName: new Map([['zero_scorer', 0]]),
        avgValuesByName: new Map([['zero_scorer', 0]]),
      });

      expect(screen.getByText('zero_scorer')).toBeInTheDocument();
      expect(screen.getByText('0')).toBeInTheDocument();
      expect(screen.getByText('0.00')).toBeInTheDocument();
    });
  });
});
