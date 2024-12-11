/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import { GetLinkModal } from './GetLinkModal';

describe('GetLinkModal', () => {
  let wrapper;
  let minimalProps: any;

  beforeEach(() => {
    minimalProps = {
      visible: true,
      onCancel: () => {},
      link: 'link',
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<GetLinkModal {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
});
