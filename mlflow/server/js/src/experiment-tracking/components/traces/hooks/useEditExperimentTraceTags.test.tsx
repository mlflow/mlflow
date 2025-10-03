import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { useEditExperimentTraceTags } from './useEditExperimentTraceTags';
import { renderWithIntl, screen, within } from '../../../../common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import { MlflowService } from '../../../sdk/MlflowService';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000); // Larger timeout for integration testing (form rendering)

const mockTraceInfo: ModelTraceInfo = {
  request_id: 'tr-test-request-id-1',
  tags: [{ key: 'existing-tag', value: 'existing-value' }],
};

describe('useEditExperimentTraceTag', () => {
  function renderTestComponent(trace: ModelTraceInfo, useV3Apis = false) {
    function TestComponent() {
      const { showEditTagsModalForTrace, EditTagsModal } = useEditExperimentTraceTags({
        onSuccess: jest.fn(),
        useV3Apis,
      });
      return (
        <>
          <button onClick={() => showEditTagsModalForTrace(trace)}>trigger button</button>
          {EditTagsModal}
        </>
      );
    }
    const { rerender } = renderWithIntl(<TestComponent />);
    return { rerender: () => rerender(<TestComponent />) };
  }

  test('it should properly add tag with key and value', async () => {
    // Mock the service functions
    jest.spyOn(MlflowService, 'setExperimentTraceTag').mockImplementation(() => Promise.resolve({}));
    jest.spyOn(MlflowService, 'deleteExperimentTraceTag').mockImplementation(() => Promise.resolve({}));

    // Render the component
    renderTestComponent(mockTraceInfo);

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
    expect(MlflowService.setExperimentTraceTag).toHaveBeenCalledTimes(1);
    expect(MlflowService.setExperimentTraceTag).toHaveBeenCalledWith('tr-test-request-id-1', 'newtag', 'newvalue');

    // We expect one existing tag to be deleted
    expect(MlflowService.deleteExperimentTraceTag).toHaveBeenCalledTimes(1);
    expect(MlflowService.deleteExperimentTraceTag).toHaveBeenCalledWith('tr-test-request-id-1', 'existing-tag');
  });

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
    expect(MlflowService.setExperimentTraceTag).toHaveBeenCalledTimes(1);
    expect(MlflowService.setExperimentTraceTag).toHaveBeenCalledWith('tr-test-request-id-1', 'newtag', 'newvalue');

    // We expect one existing tag to be deleted
    expect(MlflowService.deleteExperimentTraceTag).toHaveBeenCalledTimes(1);
    expect(MlflowService.deleteExperimentTraceTag).toHaveBeenCalledWith('tr-test-request-id-1', 'existing-tag');
  });
});
