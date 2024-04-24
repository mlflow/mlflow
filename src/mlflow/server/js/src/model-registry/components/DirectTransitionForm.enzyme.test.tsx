/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { DirectTransitionForm } from './DirectTransitionForm';
import { ACTIVE_STAGES, Stages } from '../constants';
import { Checkbox } from '@databricks/design-system';
import _ from 'lodash';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';

describe('DirectTransitionForm', () => {
  let wrapper;
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = {
      innerRef: React.createRef(),
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(<DirectTransitionForm {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render checkbox only for active stage', () => {
    const [activeStages, nonActiveStages] = _.partition(Stages, (s) => ACTIVE_STAGES.includes(s));

    activeStages.forEach((toStage) => {
      const props = { ...minimalProps, toStage };
      wrapper = mountWithIntl(<DirectTransitionForm {...props} />);
      expect(wrapper.find(Checkbox).length).toBe(1);
    });

    nonActiveStages.forEach((toStage) => {
      const props = { ...minimalProps, toStage };
      wrapper = mountWithIntl(<DirectTransitionForm {...props} />);
      expect(wrapper.find(Checkbox).length).toBe(0);
    });
  });
});
