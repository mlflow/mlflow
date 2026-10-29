import { describe, expect, it, jest } from '@jest/globals';
import { render } from '@databricks/web-shared/test-utils/render';

import { DesignSystemProvider } from '@databricks/design-system';

import { ModelIconType } from './ModelTrace.types';
import { ModelTraceExplorerIcon } from './ModelTraceExplorerIcon';
import { shouldEnableNewEvalRunsUi } from '../../../../common/utils/FeatureUtils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

const renderIconGlyphClassName = (props: Parameters<typeof ModelTraceExplorerIcon>[0]) => {
  const { container } = render(
    <DesignSystemProvider>
      <ModelTraceExplorerIcon {...props} />
    </DesignSystemProvider>,
  );
  return container.querySelector('[role="img"]')?.className;
};

describe('ModelTraceExplorerIcon', () => {
  it('flag on: agent spans use the AI icon color, like root spans', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const rootAgentGlyph = renderIconGlyphClassName({ type: ModelIconType.AGENT, isRootSpan: true });
    const plainAgentGlyph = renderIconGlyphClassName({ type: ModelIconType.AGENT });

    expect(plainAgentGlyph).toBe(rootAgentGlyph);
  });

  it('flag on: an errored agent span keeps the danger color', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const erroredGlyph = renderIconGlyphClassName({ type: ModelIconType.AGENT, hasException: true });

    expect(erroredGlyph).not.toBe(renderIconGlyphClassName({ type: ModelIconType.AGENT }));
  });

  it('flag off: agent spans keep the default icon color', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    const rootAgentGlyph = renderIconGlyphClassName({ type: ModelIconType.AGENT, isRootSpan: true });
    const plainAgentGlyph = renderIconGlyphClassName({ type: ModelIconType.AGENT });

    expect(plainAgentGlyph).not.toBe(rootAgentGlyph);
  });
});
