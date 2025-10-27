import { describe, it, expect } from '@jest/globals';
import { render, renderHook, screen } from '@testing-library/react';
import { merge } from 'lodash';
import React from 'react';

import { useGenAITracesTableConfig, GenAITracesTableConfigProvider } from './useGenAITracesTableConfig';

// Mock shouldEnableUnifiedEvalTab
jest.mock('../utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../utils/FeatureUtils')>('../utils/FeatureUtils'),
  shouldEnableRunEvaluationReviewUIWriteFeatures: jest.fn().mockReturnValue(false),
}));

describe('GenAITracesTableConfigProvider', () => {
  it('provides default values when no config is given', () => {
    const TestComponent = () => {
      const config = useGenAITracesTableConfig();
      return (
        <div data-testid="enableRunEvaluationWriteFeatures">{String(config.enableRunEvaluationWriteFeatures)}</div>
      );
    };

    render(
      <GenAITracesTableConfigProvider>
        <TestComponent />
      </GenAITracesTableConfigProvider>,
    );

    expect(screen.getByTestId('enableRunEvaluationWriteFeatures').textContent).toBe('false');
  });

  it('overrides default values when config is provided', () => {
    const TestComponent = () => {
      const config = useGenAITracesTableConfig();
      return (
        <div data-testid="enableRunEvaluationWriteFeatures">{String(config.enableRunEvaluationWriteFeatures)}</div>
      );
    };

    render(
      <GenAITracesTableConfigProvider config={{ enableRunEvaluationWriteFeatures: true }}>
        <TestComponent />
      </GenAITracesTableConfigProvider>,
    );

    expect(screen.getByTestId('enableRunEvaluationWriteFeatures').textContent).toBe('true');
  });

  it('ignores undefined values in the provided config', () => {
    const TestComponent = () => {
      const config = useGenAITracesTableConfig();
      return (
        <div data-testid="enableRunEvaluationWriteFeatures">{String(config.enableRunEvaluationWriteFeatures)}</div>
      );
    };

    render(
      <GenAITracesTableConfigProvider config={{ enableRunEvaluationWriteFeatures: undefined }}>
        <TestComponent />
      </GenAITracesTableConfigProvider>,
    );

    expect(screen.getByTestId('enableRunEvaluationWriteFeatures').textContent).toBe('false');
  });

  it('merges provided config with default config correctly', () => {
    const mockProvidedConfig = { enableRunEvaluationWriteFeatures: true };
    const expectedConfig = merge({}, { enableRunEvaluationWriteFeatures: false }, mockProvidedConfig);

    const TestComponent = () => {
      const config = useGenAITracesTableConfig();
      return (
        <div data-testid="enableRunEvaluationWriteFeatures">{String(config.enableRunEvaluationWriteFeatures)}</div>
      );
    };

    render(
      <GenAITracesTableConfigProvider config={mockProvidedConfig}>
        <TestComponent />
      </GenAITracesTableConfigProvider>,
    );

    expect(screen.getByTestId('enableRunEvaluationWriteFeatures').textContent).toBe(
      String(expectedConfig.enableRunEvaluationWriteFeatures),
    );
  });
});

describe('useGenAITracesTableConfig', () => {
  it('returns default values when used without a provider', () => {
    const { result } = renderHook(() => useGenAITracesTableConfig());

    expect(result.current.enableRunEvaluationWriteFeatures).toBe(false); // Default from getDefaultConfig()
  });
});
