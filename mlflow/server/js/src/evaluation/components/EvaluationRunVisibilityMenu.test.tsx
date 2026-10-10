import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import { EvaluationRunVisibilityMenu } from './EvaluationRunVisibilityMenu';

test('visibility menu triggers onChangeVisibility correctly', () => {
  const mockFn = jest.fn();
  const { getByText } = render(
    <EvaluationRunVisibilityMenu onChangeVisibility={mockFn} />,
  );

  fireEvent.click(getByText('Visibility Options'));
  fireEvent.click(getByText('Show all runs'));
  expect(mockFn).toHaveBeenCalledWith('show_all');
});
