import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { act, render, screen } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import {
  CustomViewDefinitionProvider,
  getCustomViewAuthoringContext,
  useOptionalCustomViewDefinition,
  type CustomView,
} from '@databricks/web-shared/model-trace-explorer/custom-view';
import { getCustomViewSpecApplier } from '../../../shared/web-shared/model-trace-explorer/custom-view/assistant/customViewSpecApplier';
import { MOCK_V3_TRACE } from '../../../shared/web-shared/model-trace-explorer/ModelTraceExplorer.test-utils';

import { ReviewQueueCustomViewRenderer } from './ReviewQueueCustomViewContent';

jest.mock('@mlflow/mlflow/src/assistant', () => ({
  __esModule: true,
  useAssistant: () => ({ canUseAssistant: false, isPanelOpen: false, openPanel: jest.fn() }),
}));

const savedView: CustomView = {
  id: 'saved-view-1',
  name: 'Reviewer view',
  label: 'Reviewer view',
  instruction: '',
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [{ id: 'root', component: 'Text', text: 'Saved view content' }],
      },
    },
  ],
  createdAtMs: 1,
};

const SelectViewProbe = ({ selectViewRef }: { selectViewRef: { current?: (id: string) => void } }) => {
  selectViewRef.current = useOptionalCustomViewDefinition()?.selectView;
  return null;
};

const renderSavedView = (definitionProps: { onPersistView?: (view: CustomView) => Promise<void> }) => {
  const selectViewRef: { current?: (id: string) => void } = {};
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={new QueryClient()}>
          <CustomViewDefinitionProvider
            views={[savedView]}
            isLoaded
            onPersistView={definitionProps.onPersistView}
            canModifyPersistedViews={Boolean(definitionProps.onPersistView)}
          >
            <ReviewQueueCustomViewRenderer modelTrace={MOCK_V3_TRACE} enableFeedback />
            <SelectViewProbe selectViewRef={selectViewRef} />
          </CustomViewDefinitionProvider>
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
  act(() => selectViewRef.current?.(savedView.id));
};

describe('ReviewQueueCustomViewRenderer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders a saved view without publishing Assistant authoring hooks from a load-only definition', async () => {
    renderSavedView({});

    expect(await screen.findByText('Saved view content')).toBeInTheDocument();
    expect(getCustomViewAuthoringContext()).toBeNull();
    expect(getCustomViewSpecApplier()).toBeNull();
  });

  it('publishes Assistant authoring hooks only when the definition can persist views', async () => {
    renderSavedView({ onPersistView: () => Promise.resolve() });

    expect(await screen.findByText('Saved view content')).toBeInTheDocument();
    expect(getCustomViewAuthoringContext()).not.toBeNull();
    expect(getCustomViewSpecApplier()).not.toBeNull();
  });
});
