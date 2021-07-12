import React from 'react';
import { EditableTagsTableView, EditableTagsTableViewImpl } from './EditableTagsTableView';
import { mountWithIntl } from '../utils/TestUtils';
import { BrowserRouter } from 'react-router-dom';

describe('unit tests', () => {
  let wrapper;
  let instance;
  const minimalProps = {
    tags: {
      tag1: { getKey: () => 'tag1', getValue: () => 'value1' },
      tag2: { getKey: () => 'tag2', getValue: () => 'value2' },
    },
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn((opts) => (c) => c) },
    handleAddTag: () => {},
    handleSaveEdit: () => {},
    handleDeleteTag: () => {},
    isRequestPending: false,
  };

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <BrowserRouter>
        <EditableTagsTableView {...minimalProps} />
      </BrowserRouter>,
    );
    expect(wrapper.length).toBe(1);
  });

  test('should validate tag name properly', () => {
    wrapper = mountWithIntl(
      <BrowserRouter>
        <EditableTagsTableView {...minimalProps} />
      </BrowserRouter>,
    );
    instance = wrapper.find(EditableTagsTableViewImpl).instance();
    const validationCallback = jest.fn();
    instance.tagNameValidator(undefined, 'tag1', validationCallback);
    expect(validationCallback).toBeCalledWith('Tag "tag1" already exists.');
  });
});
