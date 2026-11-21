import { describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { TracesV4PageContent } from './TracesV4PageContent';
import * as controllerModule from '../hooks/useTracesV4Controller';
import * as savedViewsModule from './TracesV4SavedViews';
import * as tableModule from './TracesV4TableSurface';
import { renderTracesV4Page } from '../test-utils/renderTracesV4Page';
import { EXPERIMENT_ID, URL, history, findTraceRow, state } from '../test-utils/tracesV4PageContentTestBed';

const Navigation = () => {
  const [params, setParams] = useSearchParams();
  return (
    <>
      <output data-testid="selected-span">{params.get('spanId')}</output>
      <button
        onClick={() =>
          setParams(
            (params) => {
              params.set('spanId', 'child');
              return params;
            },
            { replace: true },
          )
        }
      >
        Select child URL
      </button>
    </>
  );
};

describe('trace URL subscriptions', () => {
  test('span-only navigation does not rerender the page or saved views', async () => {
    const controller = jest.spyOn(controllerModule, 'useTracesV4Controller');
    const savedViews = jest.spyOn(savedViewsModule, 'useTracesV4SavedViews');
    const table = jest.spyOn(tableModule, 'TracesV4TableSurface');
    renderTracesV4Page({
      initialUrl: URL,
      routes: [
        {
          path: '*',
          element: (
            <>
              <Navigation />
              <TracesV4PageContent experimentId={EXPERIMENT_ID} />
            </>
          ),
        },
      ],
      history,
      dataRouter: true,
    });
    await findTraceRow('tr-000');
    await waitFor(() => expect(state.searchCalls.length).toBeGreaterThan(0));
    const pageRenders = controller.mock.calls.length;
    const savedViewRenders = savedViews.mock.calls.length;
    const tableRenders = table.mock.calls.length;
    fireEvent.click(screen.getByText('Select child URL'));
    expect(screen.getByTestId('selected-span')).toHaveTextContent('child');
    expect({
      page: controller.mock.calls.length - pageRenders,
      savedViews: savedViews.mock.calls.length - savedViewRenders,
      table: table.mock.calls.length - tableRenders,
    }).toEqual({ page: 0, savedViews: 0, table: 0 });
    controller.mockRestore();
    savedViews.mockRestore();
    table.mockRestore();
  });
});
