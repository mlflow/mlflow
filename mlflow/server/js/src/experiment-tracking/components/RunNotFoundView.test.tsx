/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { RunNotFoundView } from './RunNotFoundView';

describe('RunNotFoundView', () => {
  let minimalProps: any;
  const mockRunId = 'This is a mock run ID';

  beforeEach(() => {
    minimalProps = { runId: mockRunId };
  });

  test('should render with minimal props without exploding', () => {
    shallow(<RunNotFoundView {...minimalProps} />);
  });
});
