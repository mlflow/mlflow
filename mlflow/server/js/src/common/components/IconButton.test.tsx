import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { IconButton } from './IconButton';
import userEvent from '@testing-library/user-event';

const minimalProps = { icon: () => <span /> };

describe('IconButton', () => {
  test('should render with minimal props without exploding', () => {
    renderWithIntl(<IconButton {...minimalProps} />);
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  test('should not have padding', () => {
    renderWithIntl(<IconButton {...minimalProps} />);
    expect(screen.getByRole('button')).toHaveStyle('padding: 0px');
  });

  test('should propagate props to Button', () => {
    const props = {
      className: 'dummy-class',
      style: { margin: 5 },
    };
    renderWithIntl(<IconButton {...{ ...minimalProps, ...props }} />);

    expect(screen.getByRole('button')).toHaveStyle('padding: 0px');
    expect(screen.getByRole('button')).toHaveStyle('margin: 5px');
  });

  test('should trigger onClick when clicked', async () => {
    const mockOnClick = jest.fn();
    const props = {
      ...minimalProps,
      onClick: mockOnClick,
    };
    renderWithIntl(<IconButton {...props} />);
    await userEvent.click(screen.getByRole('button'));
    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });
});
