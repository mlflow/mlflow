import { getShareFeedbackOverflowMenuItem } from './ExperimentViewHeader.utils';

describe('getShareFeedbackOverflowMenuItem', () => {
  // minimal test to satisfy knip
  it('returns an object with the correct structure', () => {
    const result = getShareFeedbackOverflowMenuItem();
    expect(result).toBeDefined();
    expect(result.id).toBe('feedback');
  });
});
