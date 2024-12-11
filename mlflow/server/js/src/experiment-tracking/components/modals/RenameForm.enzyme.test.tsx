/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { RenameForm } from './RenameForm';

describe('Render test', () => {
  const minimalProps = {
    type: 'run',
    name: 'Test',
    visible: true,
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn((opts) => (c: any) => c) },
  };

  it('should render with minimal props without exploding', () => {
    // @ts-expect-error TS(2769): No overload matches this call.
    const wrapper = shallow(<RenameForm {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});
