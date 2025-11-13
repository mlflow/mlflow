import { jest, describe, test, expect } from '@jest/globals';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { useEditExperimentTraceTags } from './useEditExperimentTraceTags';
import { renderWithIntl, screen, within } from '../../../../common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { MlflowService } from '../../../sdk/MlflowService';
import { DesignSystemProvider } from '@databricks/design-system';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000); // Larger timeout for integration testing (form rendering)

const mockTraceInfo: ModelTraceInfoV3 = {
  trace_id: 'tr-1',
  client_request_id: 'tr-1',
  tags: { 'existing-tag': 'existing-value' },
  trace_location: {} as any,
  request_time: '0',
  state: 'OK',
  trace_metadata: {},
};

describe('useEditExperimentTraceTag', () => {
  function renderTestComponent(trace: ModelTraceInfoV3, useV3Apis = false) {
    function TestComponent() {
      const { showEditTagsModalForTrace, EditTagsModal } = useEditExperimentTraceTags({
        onSuccess: jest.fn(),
      });
      return (
        <>
          <button onClick={() => showEditTagsModalForTrace(trace)}>trigger button</button>
          {EditTagsModal}
        </>
      );
    }
    const { rerender } = renderWithIntl(
      <DesignSystemProvider>
        <TestComponent />
      </DesignSystemProvider>,
    );
    return {
      rerender: () =>
        rerender(
          <DesignSystemProvider>
            <TestComponent />
          </DesignSystemProvider>,
        ),
    };
  }

  test('it should properly add tag with key and value with v3 apis', async () => {
    // Mock the service functions
    jest.spyOn(MlflowService, 'setExperimentTraceTagV3').mockImplementation(() => Promise.resolve({}));
    jest.spyOn(MlflowService, 'deleteExperimentTraceTagV3').mockImplementation(() => Promise.resolve({}));

    // Mock v2 apis to throw an error
    jest.spyOn(MlflowService, 'setExperimentTraceTag').mockImplementation(() => {
      throw new Error('Should not be called');
    });
    jest.spyOn(MlflowService, 'deleteExperimentTraceTag').mockImplementation(() => {
      throw new Error('Should not be called');
    });

    // Render the component
    renderTestComponent(mockTraceInfo, true);

    // Click on the trigger button
    await userEvent.click(screen.getByRole('button', { name: 'trigger button' }));

    // Expect the modal to be shown
    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();

    // Fill out the form
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('combobox'));
    await userEvent.type(within(screen.getByRole('dialog')).getByRole('combobox'), 'newtag');
    await userEvent.click(screen.getByText(/Add tag "newtag"/));
    await userEvent.type(screen.getByLabelText('Value'), 'newvalue');

    // Add the tag
    await userEvent.click(screen.getByLabelText('Add tag'));

    // Remove the existing tag
    await userEvent.click(within(screen.getByRole('status', { name: 'existing-tag' })).getByRole('button'));

    // Finally, save the tags
    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    // We expect one new tag to be added
    expect(MlflowService.setExperimentTraceTagV3).toHaveBeenCalledTimes(1);
    expect(MlflowService.setExperimentTraceTagV3).toHaveBeenCalledWith('tr-1', 'newtag', 'newvalue');

    // We expect one existing tag to be deleted
    expect(MlflowService.deleteExperimentTraceTagV3).toHaveBeenCalledTimes(1);
    expect(MlflowService.deleteExperimentTraceTagV3).toHaveBeenCalledWith('tr-1', 'existing-tag');
  });
});
