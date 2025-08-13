import React from 'react';
import { render, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { NoExperimentView } from './NoExperimentView';

describe('NoExperimentView', () => {
  test('should render without exploding', () => {
    render(<NoExperimentView />);
    expect(
      screen.getByRole('heading', {
        name: /no experiments exist/i,
      }),
    ).toBeInTheDocument();
  });
});
