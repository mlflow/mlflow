import { describe, expect, test } from '@jest/globals';
import { buildAssessmentColumnDefs } from './buildAssessmentColumnDefs';

describe('buildAssessmentColumnDefs', () => {
  test('builds column defs for assessment columns with correct ids and headers', () => {
    const columns = [
      { name: 'relevance', type: 'categorical' as const },
      { name: 'score', type: 'numeric' as const },
    ];
    const defs = buildAssessmentColumnDefs(columns);

    expect(defs).toHaveLength(2);
    expect(defs[0].id).toBe('assessment:relevance');
    expect(defs[1].id).toBe('assessment:score');
    // header is a render fn returning the assessment name; invoke through a callable cast.
    expect((defs[0].header as () => string)()).toBe('relevance');
    expect((defs[1].header as () => string)()).toBe('score');
  });

  test('assigns correct column sizes', () => {
    const columns = [{ name: 'test', type: 'categorical' as const }];
    const defs = buildAssessmentColumnDefs(columns);

    expect(defs[0].size).toBe(160);
    expect(defs[0].minSize).toBe(100);
    expect(defs[0].maxSize).toBe(480);
  });
});
