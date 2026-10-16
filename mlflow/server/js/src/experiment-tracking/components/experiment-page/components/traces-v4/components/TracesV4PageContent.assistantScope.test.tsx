import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { useState, type ReactNode } from 'react';
import { TracesV4PageContent } from './TracesV4PageContent';
import { renderTracesV4Page } from '../test-utils/renderTracesV4Page';
import { EXPERIMENT_ID, URL, findTraceRow, history } from '../test-utils/tracesV4PageContentTestBed';
import { ExperimentPageLayoutProvider } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-page-tabs/ExperimentPageLayout';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';

jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: () => ({ data: { tags: [] }, refetch: () => Promise.resolve({}) }),
}));

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

let mockIsAssistantPanelOpen = false;
jest.mock('@mlflow/mlflow/src/assistant', () => {
  const actual = jest.requireActual<typeof import('@mlflow/mlflow/src/assistant')>('@mlflow/mlflow/src/assistant');
  return {
    ...actual,
    useAssistant: () => ({ ...actual.useAssistant(), isPanelOpen: mockIsAssistantPanelOpen }),
  };
});

const ExperimentPageLayoutTestFixture = ({ children }: { children: ReactNode }) => {
  const [layoutContainer, setLayoutContainer] = useState<HTMLDivElement | null>(null);
  return (
    <ExperimentPageLayoutProvider value={layoutContainer}>
      <div ref={setLayoutContainer} data-testid="experiment-page-layout">
        {children}
      </div>
    </ExperimentPageLayoutProvider>
  );
};

const renderPageInLayout = () =>
  renderTracesV4Page({
    initialUrl: URL,
    routes: [
      {
        path: '/ml/experiments/:experimentId/traces',
        element: (
          <>
            <ExperimentPageLayoutTestFixture>
              <TracesV4PageContent experimentId={EXPERIMENT_ID} />
            </ExperimentPageLayoutTestFixture>
            <div data-assistant-ui="true">
              <button type="button">Assistant control</button>
            </div>
          </>
        ),
      },
    ],
    history,
    experimentId: EXPERIMENT_ID,
  });

describe('TracesV4PageContent trace drawer with the MLflow Assistant', () => {
  afterEach(() => {
    mockIsAssistantPanelOpen = false;
  });

  test('flag on, Assistant closed: opens as a standard resizable drawer over the page', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPageInLayout();

    await user.click(await findTraceRow('tr-000'));

    const drawer = await screen.findByRole('dialog');
    expect(drawer).toHaveAttribute('data-trace-drawer-fullscreen', 'false');
    expect(screen.getByTestId('trace-drawer-portal')).toHaveStyle({
      top: '0px',
      left: '0px',
      width: `${window.innerWidth}px`,
      height: `${window.innerHeight}px`,
    });
    expect(screen.getByRole('button', { name: 'Full screen' })).toBeInTheDocument();

    // Opening the Assistant from outside the drawer does not close the drawer. The drawer hides the rest of the
    // page from the accessibility tree, but the Assistant button stays clickable.
    await user.click(screen.getByRole('button', { name: 'Assistant control', hidden: true }));
    expect(drawer).toBeInTheDocument();
  }, 40000);

  test('flag on, Assistant open: fills only the experiment page, and can exit full screen', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    mockIsAssistantPanelOpen = true;
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPageInLayout();

    const drawerScope = await screen.findByTestId('experiment-page-layout');
    jest.spyOn(drawerScope, 'getBoundingClientRect').mockReturnValue({
      x: 120,
      y: 40,
      top: 40,
      right: 820,
      bottom: 640,
      left: 120,
      width: 700,
      height: 600,
      toJSON: () => ({}),
    });

    await user.click(await findTraceRow('tr-000'));
    const drawer = await screen.findByRole('dialog');
    expect(screen.getByTestId('trace-drawer-portal')).toHaveStyle({
      top: '40px',
      left: '120px',
      width: '700px',
      height: '600px',
    });
    expect(drawer).toHaveAttribute('data-trace-drawer-fullscreen', 'true');
    expect(screen.getByRole('button', { name: 'Exit full screen' })).toBeInTheDocument();
    expect(screen.queryByRole('separator', { name: 'Resize trace drawer' })).not.toBeInTheDocument();
    expect(within(drawer).queryByText('Share')).not.toBeInTheDocument();
    expect(within(drawer).getByRole('button', { name: 'Share' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Exit full screen' }));
    expect(drawer).toHaveAttribute('data-trace-drawer-fullscreen', 'false');
    expect(screen.getByRole('button', { name: 'Full screen' })).toBeInTheDocument();
    expect(drawerScope).toContainElement(screen.getByRole('separator', { name: 'Resize trace drawer' }));

    // Working in the Assistant does not close the drawer.
    await user.click(screen.getByRole('button', { name: 'Assistant control' }));
    expect(drawer).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Close trace panel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  }, 40000);

  test('flag off, Assistant open: keeps the existing assistant-aware drawer', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    mockIsAssistantPanelOpen = true;
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPageInLayout();

    await user.click(await findTraceRow('tr-000'));
    const drawer = await screen.findByRole('dialog');

    expect(screen.queryByTestId('trace-drawer-portal')).not.toBeInTheDocument();
    expect(screen.queryByTestId('traces-v4-drawer-scope')).not.toBeInTheDocument();
    expect(drawer).not.toHaveAttribute('data-trace-drawer-fullscreen');
    expect(screen.getByRole('button', { name: 'Full screen' })).toBeInTheDocument();
  }, 40000);
});
