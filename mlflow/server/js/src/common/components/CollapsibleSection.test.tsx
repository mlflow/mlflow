import React from 'react';
import userEvent from '@testing-library/user-event';

import { CollapsibleSection } from './CollapsibleSection';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

describe('CollapsibleSection', () => {
  let wrapper;
  let minimalProps: {
    title: string | any;
    children: React.ReactNode;
    forceOpen?: boolean;
  };

  beforeEach(() => {
    minimalProps = {
      title: 'testTitle',
      children: 'testChild',
    };
  });

  test('should render in initial collapsed state', () => {
    wrapper = renderWithIntl(<CollapsibleSection {...minimalProps} defaultCollapsed />);
    expect(wrapper.getByRole('button')).toHaveTextContent('testTitle');
    expect(wrapper.getByRole('button')).toHaveAttribute('aria-expanded', 'false');
    expect(wrapper.container).not.toHaveTextContent('testChild');
  });

  test('should render in initial expanded state', () => {
    wrapper = renderWithIntl(<CollapsibleSection {...minimalProps} />);
    expect(wrapper.container).toHaveTextContent('testChild');
  });

  test('should expand when clicked', async () => {
    wrapper = renderWithIntl(<CollapsibleSection {...minimalProps} defaultCollapsed />);
    expect(wrapper.container).not.toHaveTextContent('testChild');
    expect(wrapper.getByRole('button')).toHaveAttribute('aria-expanded', 'false');
    await userEvent.click(wrapper.getByRole('button'));
    expect(wrapper.container).toHaveTextContent('testChild');
    expect(wrapper.getByRole('button')).toHaveAttribute('aria-expanded', 'true');
  });
});
