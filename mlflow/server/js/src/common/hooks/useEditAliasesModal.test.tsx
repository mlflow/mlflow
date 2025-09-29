import userEventGlobal, { PointerEventsCheckLevel } from '@testing-library/user-event';

import { useEditAliasesModal } from './useEditAliasesModal';
import type { ModelEntity } from '../../experiment-tracking/types';
import {
  fastFillInput,
  findAntdOptionContaining,
  renderWithIntl,
  type RenderResult,
} from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ErrorWrapper } from '../utils/ErrorWrapper';

const MOCK_MODEL = {
  name: 'test-model',
  aliases: [
    { alias: 'champion', version: '1' },
    { alias: 'challenger', version: '2' },
    { alias: 'latest', version: '2' },
  ],
};

const MOCK_MODEL_MANY_ALIASES = {
  name: 'test-model',
  aliases: [
    { alias: 'champion1', version: '1' },
    { alias: 'champion2', version: '1' },
    { alias: 'champion3', version: '1' },
    { alias: 'champion4', version: '1' },
    { alias: 'champion5', version: '1' },
    { alias: 'champion6', version: '1' },
    { alias: 'champion7', version: '1' },
    { alias: 'champion8', version: '1' },
    { alias: 'champion9', version: '1' },
    { alias: 'champion10', version: '1' },
    { alias: 'challenger', version: '2' },
  ],
};

const handleSave = jest.fn(() => new Promise((resolve) => resolve));

describe('useEditAliasesModal', () => {
  let userEvent: ReturnType<typeof userEventGlobal.setup>;

  beforeEach(() => {
    // Remove pointer event check otherwise there's some div that and pointer-events: none that blocks clicks
    userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const renderComponentWithHook = (model: Partial<ModelEntity> = {}, onSuccess = () => {}) => {
    const TestComponent = () => {
      const { EditAliasesModal, showEditAliasesModal } = useEditAliasesModal({
        aliases: model.aliases ?? [],
        onSuccess,
        onSave: handleSave,
        getTitle: (version: string) => <h1>Title</h1>,
        description: <p>Description</p>,
      });
      return (
        <div>
          <button onClick={() => showEditAliasesModal('1')}>edit version 1 aliases</button>
          <button onClick={() => showEditAliasesModal('2')}>edit version 2 aliases</button>
          {EditAliasesModal}
        </div>
      );
    };

    return renderWithIntl(<TestComponent />);
  };

  const findOption = (scope: RenderResult, text: string) =>
    scope.getAllByTestId('model-alias-option').find(({ textContent }) => textContent?.includes(text)) as HTMLElement;

  test('should initialize and render modal with properly displayed tags', async () => {
    const component = renderComponentWithHook(MOCK_MODEL);

    await userEvent.click(component.getByText('edit version 1 aliases'));

    expect(component.getByRole('status', { name: 'champion' })).toBeInTheDocument();

    await userEvent.click(component.getByText('Cancel'));

    await userEvent.click(component.getByText('edit version 2 aliases'));

    expect(component.getByRole('status', { name: 'challenger' })).toBeInTheDocument();
    expect(component.getByRole('status', { name: 'latest' })).toBeInTheDocument();
  });

  test('should display warning for conflicting aliases', async () => {
    const component = renderComponentWithHook(MOCK_MODEL);

    await userEvent.click(component.getByText('edit version 1 aliases'));

    expect(component.getByTitle('champion')).toBeInTheDocument();

    await userEvent.click(component.getByRole('combobox'));

    await userEvent.type(component.getByRole('combobox'), 'challenger');
    await userEvent.click(await findOption(component, 'challenger'));

    expect(
      component.getByText(
        'The "challenger" alias is also being used on version 2. Adding it to this version will remove it from version 2.',
      ),
    ).toBeInTheDocument();
  });

  test('should select a new alias', async () => {
    const component = renderComponentWithHook(MOCK_MODEL);

    await userEvent.click(component.getByText('edit version 1 aliases'));

    expect(component.getByTitle('champion')).toBeInTheDocument();

    await userEvent.click(component.getByRole('combobox'));

    await fastFillInput(component.getByRole('combobox') as HTMLInputElement, 'new_alias_for_v1');
    await userEvent.click(await findAntdOptionContaining('new_alias_for_v1'));

    expect(component.getByRole('status', { name: 'champion' })).toBeInTheDocument();
    expect(component.getByRole('status', { name: 'new_alias_for_v1' })).toBeInTheDocument();
  });

  test('should not be able to add too many aliases', async () => {
    const component = renderComponentWithHook(MOCK_MODEL_MANY_ALIASES);

    await userEvent.click(component.getByText('edit version 1 aliases'));
    await userEvent.click(component.getByRole('combobox'));
    await userEvent.click(findOption(component, 'challenger'));

    expect(component.getByText(/You are exceeding a limit of \d+ aliases/)).toBeInTheDocument();
    expect(component.getByRole('button', { name: 'Save aliases' })).toBeDisabled();
  });

  test('should invoke proper API requests for adding and removing aliases', async () => {
    const component = renderComponentWithHook(MOCK_MODEL);

    await userEvent.click(component.getByText('edit version 1 aliases'));

    expect(component.getByTitle('champion')).toBeInTheDocument();

    expect(component.getByRole('button', { name: 'Save aliases' })).toBeDisabled();

    await userEvent.click(component.getByRole('combobox'));
    await userEvent.click(findOption(component, 'champion'));
    await userEvent.click(findOption(component, 'challenger'));

    await userEvent.click(component.getByText('Save aliases'));

    expect(handleSave).toHaveBeenCalledWith('1', ['champion'], ['challenger']);
  });

  test('should display error message on failure', async () => {
    handleSave.mockRejectedValue(new ErrorWrapper({ message: 'some error message' }, 500));

    const component = renderComponentWithHook(MOCK_MODEL);

    await userEvent.click(component.getByText('edit version 1 aliases'));

    await userEvent.click(component.getByRole('combobox'));
    await userEvent.click(findOption(component, 'challenger'));

    await userEvent.click(component.getByText('Save aliases'));

    expect(await component.findByText(/some error message/)).toBeInTheDocument();
  });
});
