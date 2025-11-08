import userEvent from '@testing-library/user-event';
import { KeyValueTagFullViewModal } from './KeyValueTagFullViewModal';
import { screen, renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

describe('KeyValueTagFullViewModal', () => {
  const mockSetIsKeyValueTagFullViewModalVisible = jest.fn();

  let navigatorClipboard: Clipboard;

  // Prepare fake clipboard
  beforeAll(() => {
    navigatorClipboard = navigator.clipboard;
    (navigator.clipboard as any) = { writeText: jest.fn() };
  });

  // Cleanup and restore clipboard
  afterAll(() => {
    (navigator.clipboard as any) = navigatorClipboard;
  });

  test('renders the component', async () => {
    const longKey = '123'.repeat(100);
    const longValue = 'abc'.repeat(100);

    renderWithIntl(
      <KeyValueTagFullViewModal
        tagKey={longKey}
        tagValue={longValue}
        setIsKeyValueTagFullViewModalVisible={mockSetIsKeyValueTagFullViewModalVisible}
        isKeyValueTagFullViewModalVisible
      />,
    );

    expect(screen.getByText('Tag: ' + longKey)).toBeInTheDocument();
    expect(screen.getByText(longValue)).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText('Copy'));

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(longValue);

    await userEvent.click(screen.getByLabelText('Close'));

    expect(mockSetIsKeyValueTagFullViewModalVisible).toHaveBeenCalledWith(false);
  });
});
