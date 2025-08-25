import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import ExpandableList from './ExpandableList';

const minimalProps = {
  // eslint-disable-next-line react/jsx-key
  children: [<div>testchild</div>],
};

const advancedProps = {
  // eslint-disable-next-line react/jsx-key
  children: [<div>testchild1</div>, <div>testchild2</div>],
  showLines: 1,
};

describe('ExpandableList', () => {
  test('should render with minimal props without exploding', () => {
    renderWithIntl(<ExpandableList {...minimalProps} />);
    expect(screen.getByText('testchild')).toBeInTheDocument();
  });

  test('expanding a longer list displays single element and expander and correctly expands', async () => {
    renderWithIntl(<ExpandableList {...advancedProps} />);
    expect(screen.getByText('testchild1')).toBeInTheDocument();
    expect(screen.queryByText('testchild2')).not.toBeInTheDocument();
    expect(screen.getByText('+1 more')).toBeInTheDocument();

    await userEvent.click(screen.getByText('+1 more'));
    expect(screen.getByText('testchild1')).toBeInTheDocument();
    expect(screen.getByText('testchild2')).toBeInTheDocument();
    expect(screen.getByText('Less')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Less'));
    expect(screen.getByText('testchild1')).toBeInTheDocument();
    expect(screen.queryByText('testchild2')).not.toBeInTheDocument();
    expect(screen.getByText('+1 more')).toBeInTheDocument();
  });
});
