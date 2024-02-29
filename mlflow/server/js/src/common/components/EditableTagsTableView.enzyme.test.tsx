/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { EditableTagsTableView, EditableTagsTableViewImpl } from './EditableTagsTableView';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { BrowserRouter } from '../../common/utils/RoutingUtils';
import { DesignSystemProvider } from '@databricks/design-system';

describe('unit tests', () => {
  let wrapper;
  let instance;
  const minimalProps = {
    tags: {
      tag1: { getKey: () => 'tag1', getValue: () => 'value1' },
      tag2: { getKey: () => 'tag2', getValue: () => 'value2' },
    },
    // eslint-disable-next-line no-unused-vars
    form: { getFieldDecorator: jest.fn((opts) => (c: any) => c) },
    handleAddTag: () => {},
    handleSaveEdit: () => {},
    handleDeleteTag: () => {},
    isRequestPending: false,
  };

  const createComponentInstance = () =>
    mountWithIntl(
      <DesignSystemProvider>
        <BrowserRouter>
          <EditableTagsTableView {...minimalProps} />
        </BrowserRouter>
      </DesignSystemProvider>,
    );

  test('should render with minimal props without exploding', () => {
    wrapper = createComponentInstance();
    expect(wrapper.length).toBe(1);
  });

  test('should validate tag name properly', () => {
    wrapper = createComponentInstance();
    instance = wrapper.find(EditableTagsTableViewImpl).instance();
    const validationCallback = jest.fn();
    instance.tagNameValidator(undefined, 'tag1', validationCallback);
    expect(validationCallback).toBeCalledWith('Tag "tag1" already exists.');
  });
});
