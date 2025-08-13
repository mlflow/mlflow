import React from 'react';
import { TrimmedText } from './TrimmedText';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

const trimmedTextDataTestId = 'trimmed-text';
const trimmedTextButtonDataTestId = 'trimmed-text-button';

const getDefaultTrimmedTextProps = (overrides = {}) => ({
  text: '0123456789',
  maxSize: 10,
  className: 'some class',
  allowShowMore: false,
  dataTestId: trimmedTextDataTestId,
  ...overrides,
});

describe('TrimmedText', () => {
  test.each([true, false])(
    'render normal text if length is less than or equal to max size when allowShowMore is %s',
    (allowShowMore) => {
      renderWithIntl(<TrimmedText {...getDefaultTrimmedTextProps({ allowShowMore: allowShowMore })} />);
      expect(screen.getByTestId(trimmedTextDataTestId)).toHaveTextContent('0123456789');
    },
  );

  test('render trimmed text if length is greater than max size', () => {
    renderWithIntl(<TrimmedText {...getDefaultTrimmedTextProps({ maxSize: 5 })} />);
    expect(screen.getByTestId(trimmedTextDataTestId)).toHaveTextContent('01234...');
    expect(screen.queryByTestId(trimmedTextButtonDataTestId)).not.toBeInTheDocument();
  });

  test('render show more button if configured', async () => {
    renderWithIntl(<TrimmedText {...getDefaultTrimmedTextProps({ maxSize: 5, allowShowMore: true })} />);

    const trimmedText = screen.getByTestId(trimmedTextDataTestId);
    const button = screen.getByTestId(trimmedTextButtonDataTestId);

    expect(trimmedText).toHaveTextContent('01234...');
    expect(button).toBeInTheDocument();
    expect(button).toHaveTextContent('expand');

    await userEvent.click(button);

    expect(trimmedText).toHaveTextContent('0123456789');
    expect(button).toHaveTextContent('collapse');

    await userEvent.click(button);

    expect(trimmedText).toHaveTextContent('01234...');
    expect(button).toHaveTextContent('expand');
  });
});
