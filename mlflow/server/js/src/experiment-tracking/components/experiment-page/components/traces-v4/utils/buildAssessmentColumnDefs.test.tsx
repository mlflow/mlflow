import { describe, expect, test } from '@jest/globals';
import { render as rtlRender } from '@testing-library/react';
import { buildAssessmentColumnDefs } from './buildAssessmentColumnDefs';

describe('buildAssessmentColumnDefs', () => {
  test('builds column defs for assessment columns with correct ids and labelText', () => {
    const columns = [
      { name: 'relevance', type: 'categorical' as const },
      { name: 'score', type: 'numeric' as const },
    ];
    const defs = buildAssessmentColumnDefs(columns);

    expect(defs).toHaveLength(2);
    expect(defs[0].id).toBe('assessment:relevance');
    expect(defs[1].id).toBe('assessment:score');
    // labelText is set for a11y on the column menu trigger.
    expect(defs[0].labelText).toBe('relevance');
    expect(defs[1].labelText).toBe('score');
  });

  test('renders assessment column header with gavel icon', () => {
    const columns = [{ name: 'relevance', type: 'categorical' as const }];
    const defs = buildAssessmentColumnDefs(columns);

    const headerElement = defs[0].header as () => JSX.Element;
    const { container } = rtlRender(headerElement());

    expect(container.querySelector('[data-testid="trace-assessment-column-icon"]')).toBeInTheDocument();
    expect(container.textContent).toContain('relevance');
  });

  test('assigns correct column sizes', () => {
    const columns = [{ name: 'test', type: 'categorical' as const }];
    const defs = buildAssessmentColumnDefs(columns);

    expect(defs[0].size).toBe(160);
    expect(defs[0].minSize).toBe(100);
    expect(defs[0].maxSize).toBe(480);
  });
});
