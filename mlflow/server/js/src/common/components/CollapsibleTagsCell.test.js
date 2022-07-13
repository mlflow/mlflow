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

  test('should not show toggle link for NUM_TAGS_ON_COLLAPSED tags', () => {
    // Assume we have exactly `NUM_TAGS_ON_COLLAPSED` tags
    const props = setupProps(NUM_TAGS_ON_COLLAPSED);
    wrapper = shallow(<CollapsibleTagsCell {...props} />);
    instance = wrapper.instance();
    expect(wrapper.find('.expander-test')).toHaveLength(0);
  });

  test('should not show toggle link for NUM_TAGS_ON_COLLAPSED + 1 tags', () => {
    // Assume we have exactly `NUM_TAGS_ON_COLLAPSED` + 1 tags
    const props = setupProps(NUM_TAGS_ON_COLLAPSED + 1);
    wrapper = shallow(<CollapsibleTagsCell {...props} />);
    instance = wrapper.instance();
    expect(wrapper.find('.expander-text')).toHaveLength(0);
  });

  test('toggle link should work as expected', () => {
    const numTags = NUM_TAGS_ON_COLLAPSED + 2;
    const props = setupProps(numTags);
    wrapper = shallow(<CollapsibleTagsCell {...props} />);
    instance = wrapper.instance();
    expect(wrapper.find('.tag-cell-item')).toHaveLength(NUM_TAGS_ON_COLLAPSED);
    expect(wrapper.find('.expander-text')).toHaveLength(1);
    instance.setState({ collapsed: false });
    expect(wrapper.find('.tag-cell-item')).toHaveLength(numTags);
  });

  test('tooltip should contain tag name and value', () => {
    const numTags = 1;
    const props = setupProps(numTags);
    wrapper = shallow(<CollapsibleTagsCell {...props} />);
    instance = wrapper.instance();
    expect(wrapper.find('.tag-cell-item')).toHaveLength(1);
    const tooltip = wrapper.find('Tooltip');
    expect(tooltip).toHaveLength(1);
    expect(tooltip.props().title).toBe('tag0: value0');
  });
});
