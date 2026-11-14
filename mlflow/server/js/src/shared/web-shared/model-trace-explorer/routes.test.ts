import { describe, it, expect } from '@jest/globals';

import { getExperimentPagePlaygroundRoute } from './routes';

describe('getExperimentPagePlaygroundRoute', () => {
  it('builds the bare playground route with no trace context', () => {
    expect(getExperimentPagePlaygroundRoute('123')).toBe('/experiments/123/playground');
  });

  it('appends the trace id for the trace-level "Open in Playground"', () => {
    expect(getExperimentPagePlaygroundRoute('123', { traceId: 'tr-1' })).toBe(
      '/experiments/123/playground?traceId=tr-1',
    );
  });

  it('appends both the trace id and span id for the span-level "Open in Playground"', () => {
    expect(getExperimentPagePlaygroundRoute('123', { traceId: 'tr-1', spanId: 'span-9' })).toBe(
      '/experiments/123/playground?traceId=tr-1&spanId=span-9',
    );
  });
});
