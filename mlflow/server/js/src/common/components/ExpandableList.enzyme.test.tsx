/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import ExpandableList from './ExpandableList';

describe('ExpandableList', () => {
  let wrapper;
  let minimalProps: any;
  let advancedProps: any;

  beforeEach(() => {
    minimalProps = {
      children: [<div className="minimal-prop">testchild</div>],
    };

    advancedProps = {
      children: [<div className="minimal-prop-1">testchild1</div>, <div className="minimal-prop-2">testchild2</div>],
      showLines: 1,
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ExpandableList {...minimalProps} />);
    expect(wrapper.find('.minimal-prop')).toHaveLength(1);
    expect(wrapper.find('.expandable-list-item')).toHaveLength(1);
  });

  test('expanding a longer list displays single element and expander and correctly expands', () => {
    wrapper = shallow(<ExpandableList {...advancedProps} />);
    expect(wrapper.find('.expandable-list-item')).toHaveLength(1);
    expect(wrapper.find('.expander-text')).toHaveLength(1);
    const expander = wrapper.find('.expander-text').at(0);
    expander.simulate('click');
    expect(wrapper.find('.minimal-prop-1')).toHaveLength(1);
    expect(wrapper.find('.minimal-prop-2')).toHaveLength(1);
    expect(wrapper.find('.expander-text')).toHaveLength(1);
  });
});
