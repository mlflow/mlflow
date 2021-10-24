import React from 'react';
import { shallow } from 'enzyme';
import { CollapsibleTagsCell, NUM_TAGS_ON_COLLAPSED } from './CollapsibleTagsCell';
import _ from 'lodash';

describe('unit tests', () => {
  let wrapper;
  let instance;
  let minimalProps;

  const setupProps = (numTags) => {
    const tags = {};
    _.range(numTags).forEach((n) => {
      tags[`tag${n}`] = { getKey: () => `tag${n}`, getValue: () => `value${n}` };
    });
    return { tags, onToggle: jest.fn() };
  };

  beforeEach(() => {
    minimalProps = setupProps(NUM_TAGS_ON_COLLAPSED + 1);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<CollapsibleTagsCell {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should not show toggle link for less than NUM_TAGS_ON_COLLAPSED tags', () => {
    // Assume we have exactly `NUM_TAGS_ON_COLLAPSED` tags
    const props = setupProps(NUM_TAGS_ON_COLLAPSED);
    wrapper = shallow(<CollapsibleTagsCell {...props} />);
    instance = wrapper.instance();
    expect(wrapper.find('.tag-cell-toggle-link')).toHaveLength(0);
  });

  test('toggle link should work as expected', () => {
    const numTags = NUM_TAGS_ON_COLLAPSED + 2;
    const props = setupProps(numTags);
    wrapper = shallow(<CollapsibleTagsCell {...props} />);
    instance = wrapper.instance();
    expect(wrapper.find('.tag-cell-item')).toHaveLength(NUM_TAGS_ON_COLLAPSED);
    instance.setState({ collapsed: false });
    expect(wrapper.find('.tag-cell-item')).toHaveLength(numTags);
  });
});
