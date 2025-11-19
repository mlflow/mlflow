import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import { EvaluationRunVisibilityMenu } from './EvaluationRunVisibilityMenu';

describe('EvaluationRunVisibilityMenu', () => {
  test('calls onChangeVisibility correctly', () => {
    const mockFn = jest.fn();
    const { getByText } = render(<EvaluationRunVisibilityMenu onChangeVisibility={mockFn} />);

    fireEvent.click(getByText('Visibility Options'));
    fireEvent.click(getByText('Show all runs'));

    expect(mockFn).toHaveBeenCalledWith('show_all');
  });
});
