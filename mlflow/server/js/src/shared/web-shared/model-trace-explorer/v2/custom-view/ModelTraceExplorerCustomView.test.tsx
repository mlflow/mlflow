import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { ModelTraceExplorerCustomView } from './ModelTraceExplorerCustomView';
import { ModelTraceExplorerCustomViewRenderer } from './ModelTraceExplorerCustomViewRenderer';
import { ModelTraceExplorerCustomViewSelector } from '../ModelTraceExplorerCustomViewSelector';
import {
  useCustomViewAssistantBridge,
  type CustomViewAssistantBridge,
} from '../../custom-view/assistant/useCustomViewAssistantBridge';
import {
  CustomViewDefinitionProvider,
  useOptionalCustomViewDefinition,
} from '../../custom-view/CustomViewDefinitionContext';
import type { CustomView } from '../../custom-view/customViewDefinition';
import type { AgentTraceData } from '../../custom-view/agent/buildAgentPrompt';
import type { CreateAssessmentPayload } from '../../api';
import type { ModelTrace } from '../ModelTrace.types';
import { MOCK_TRACE, MOCK_V3_TRACE } from '../../ModelTraceExplorer.test-utils';
import { setupServer } from '../../../test-utils/setup-msw';

// Drive the component off the bridge's return value, without the module-level
// Assistant registries the real bridge wires up.
jest.mock('../../custom-view/assistant/useCustomViewAssistantBridge', () => ({
  useCustomViewAssistantBridge: jest.fn(),
}));

// Route assessment writes through the V3 endpoint so the tests can assert on the
// request bodies, and pin the assessment author.
jest.mock('../../FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../FeatureUtils')>('../../FeatureUtils'),
  shouldUseTracesV4API: jest.fn(() => false),
}));

jest.mock('../../../global-settings/getUser', () => ({
  ...jest.requireActual<typeof import('../../../global-settings/getUser')>('../../../global-settings/getUser'),
  getUser: jest.fn(() => 'test-user'),
}));

const mockUseCustomViewAssistantBridge = jest.mocked(useCustomViewAssistantBridge);

const modelTraceInfo = { request_id: 'tr-test' } as ModelTrace['info'];

const makeView = (id: string, overrides: Partial<CustomView> = {}): CustomView => ({
  id,
  name: `name-${id}`,
  label: `label-${id}`,
  instruction: `do ${id}`,
  template: [
    {
      version: 'v0.9',
      updateComponents: { surfaceId: 'main', components: [{ id: 'root', component: 'Text', text: 'hi' }] },
    },
  ],
  createdAtMs: 1,
  ...overrides,
});

const persistedView = makeView('v1');
const rootSpanFieldView = makeView('root-span-field', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          {
            id: 'root',
            component: 'Text',
            text: { $source: 'spanField', spanRef: 'root', field: 'inputs', path: ['query'] },
          },
        ],
      },
    },
  ],
});
const v3RootSpanFieldView = makeView('v3-root-span-field', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          {
            id: 'root',
            component: 'Text',
            text: { $source: 'spanField', spanRef: 'root', field: 'inputs' },
          },
        ],
      },
    },
  ],
});

const traceFormFeedbackView = makeView('trace-form', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          { id: 'root', component: 'Column', children: ['trace-rating', 'trace-submit'] },
          {
            id: 'trace-rating',
            component: 'RadioGroup',
            name: 'TraceQuality',
            formId: 'trace',
            options: [{ label: 'Trace good', value: 'good' }],
          },
          { id: 'trace-submit', component: 'FeedbackSubmit', label: 'Submit trace feedback', formId: 'trace' },
        ],
      },
    },
  ],
});

const immediateFeedbackView = makeView('immediate-feedback', {
  template: [
    {
      version: 'v0.9',
      updateComponents: {
        surfaceId: 'main',
        components: [
          {
            id: 'root',
            component: 'FeedbackThumbsUpDownButtons',
            label: 'Was this helpful?',
            name: 'Helpfulness',
          },
        ],
      },
    },
  ],
});

const setBridge = (overrides: Partial<CustomViewAssistantBridge> = {}): CustomViewAssistantBridge => {
  const value: CustomViewAssistantBridge = {
    isAvailable: true,
    openAssistant: jest.fn(),
    isStreaming: false,
    applyError: undefined,
    clearApplyError: jest.fn(),
    ...overrides,
  } as CustomViewAssistantBridge;
  mockUseCustomViewAssistantBridge.mockReturnValue(value);
  return value;
};

const CustomViewTestHost = ({
  traceInfo,
  enableFeedback,
}: {
  traceInfo: ModelTrace['info'];
  enableFeedback: boolean;
}) => {
  const customViewDefinition = useOptionalCustomViewDefinition();
  useCustomViewAssistantBridge({ data: {} as AgentTraceData, onSpec: () => {} });
  const [displayMode, setDisplayMode] = React.useState<'default' | 'custom'>('default');
  if (!customViewDefinition) {
    return <ModelTraceExplorerCustomView modelTraceInfo={traceInfo} enableFeedback={enableFeedback} />;
  }
  return (
    <>
      <ModelTraceExplorerCustomViewSelector value={displayMode} onValueChange={setDisplayMode} isCustomViewEnabled />
      <ModelTraceExplorerCustomView modelTraceInfo={traceInfo} enableFeedback={enableFeedback} />
    </>
  );
};

const SelectionOnlySelectorHost = () => {
  const [displayMode, setDisplayMode] = React.useState<'default' | 'custom'>('default');
  return (
    <ModelTraceExplorerCustomViewSelector
      value={displayMode}
      onValueChange={setDisplayMode}
      isCustomViewEnabled
      canCreateCustomView
    />
  );
};

const SelectViewProbe = ({ selectViewRef }: { selectViewRef: { current?: (id: string) => void } }) => {
  const cv = useOptionalCustomViewDefinition();
  selectViewRef.current = cv?.selectView;
  return null;
};

const withProviders = (children: React.ReactNode, views: CustomView[]) => (
  <IntlProvider locale="en" messages={{}}>
    <DesignSystemProvider>
      <QueryClientProvider client={new QueryClient()}>
        <CustomViewDefinitionProvider views={views} isLoaded>
          {children}
        </CustomViewDefinitionProvider>
      </QueryClientProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

const renderWithViews = (views: CustomView[], enableFeedback: boolean) =>
  render(withProviders(<CustomViewTestHost traceInfo={modelTraceInfo} enableFeedback={enableFeedback} />, views));

const renderStandaloneCustomViewRenderer = (view: CustomView, modelTrace: ModelTrace) => {
  const selectViewRef: { current?: (id: string) => void } = {};
  render(
    withProviders(
      <>
        <ModelTraceExplorerCustomViewRenderer modelTrace={modelTrace} enableFeedback={false} />
        <SelectViewProbe selectViewRef={selectViewRef} />
      </>,
      [view],
    ),
  );
  act(() => selectViewRef.current?.(view.id));
};

describe('ModelTraceExplorerCustomView in a read-only host', () => {
  const { server } = setupServer();

  const captureAssessmentPosts = (): CreateAssessmentPayload['assessment'][] => {
    const posted: CreateAssessmentPayload['assessment'][] = [];
    server.use(
      rest.post('*/ajax-api/*/mlflow/traces/*/assessments', async (req, res, ctx) => {
        const body = (await req.json()) as CreateAssessmentPayload;
        posted.push(body.assessment);
        return res(ctx.json({ assessment: { ...body.assessment, assessment_id: 'a-new' } }));
      }),
    );
    return posted;
  };

  const selectView = async (viewName: RegExp) => {
    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: viewName }));
  };

  beforeEach(() => {
    setBridge();
  });

  it('does not stage or submit feedback when feedback is disabled by the host', async () => {
    const posted = captureAssessmentPosts();
    renderWithViews([traceFormFeedbackView], false);
    await selectView(/name-trace-form/);

    await userEvent.click(await screen.findByRole('radio', { name: 'Trace good' }));

    expect(screen.getByRole('button', { name: 'Submit trace feedback' })).toBeDisabled();
    expect(posted).toHaveLength(0);
  });

  it('stages feedback when the host enables it', async () => {
    renderWithViews([traceFormFeedbackView], true);
    await selectView(/name-trace-form/);

    await userEvent.click(await screen.findByRole('radio', { name: 'Trace good' }));

    expect(screen.getByRole('button', { name: 'Submit trace feedback' })).toBeEnabled();
  });

  it('does not submit immediate thumbs feedback when feedback is disabled by the host', async () => {
    const posted = captureAssessmentPosts();
    renderWithViews([immediateFeedbackView], false);
    await selectView(/name-immediate-feedback/);

    await userEvent.click(await screen.findByRole('button', { name: 'Thumbs up' }));

    expect(posted).toHaveLength(0);
  });

  it('hides creation without a callback while allowing a saved view to become active', async () => {
    render(withProviders(<SelectionOnlySelectorHost />, [persistedView]));

    await userEvent.click(screen.getByRole('button', { name: /Default view/ }));
    expect(screen.queryByRole('menuitem', { name: 'Create custom view' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('menuitemradio', { name: 'name-v1' }));
    await waitFor(() => expect(screen.getByRole('button', { name: /name-v1/ })).toBeInTheDocument());
  });

  it('does not render an empty toolbar row above a saved view', async () => {
    renderStandaloneCustomViewRenderer(rootSpanFieldView, MOCK_TRACE);

    const title = await screen.findByRole('heading', { level: 2 });
    const scroller = title.parentElement?.parentElement;
    const customViewRoot = scroller?.parentElement;
    expect(customViewRoot?.firstElementChild).toBe(scroller);
  });

  it('renders a saved root-span field binding from the full model trace', async () => {
    renderStandaloneCustomViewRenderer(rootSpanFieldView, MOCK_TRACE);

    expect(await screen.findByText('document-qa-chain-input')).toBeInTheDocument();
  });

  it('renders a saved root-span field binding from a V3 model trace', async () => {
    renderStandaloneCustomViewRenderer(v3RootSpanFieldView, MOCK_V3_TRACE);

    expect(await screen.findByText(/document-qa-chain-input/)).toBeInTheDocument();
  });
});
