import React from 'react';
import { shallow } from 'enzyme';
import { RunNotFoundView } from './RunNotFoundView';

describe('RunNotFoundView', () => {
  let minimalProps;
  const mockRunId = 'This is a mock run ID';

  beforeEach(() => {
    minimalProps = { runId: mockRunId };
  });

  test('should render with minimal props without exploding', () => {
    shallow(<RunNotFoundView {...minimalProps} />);
  });
});
