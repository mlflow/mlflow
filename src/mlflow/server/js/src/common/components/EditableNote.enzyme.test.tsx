/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { EditableNote, EditableNoteImpl } from './EditableNote';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { BrowserRouter } from '../../common/utils/RoutingUtils';

describe('EditableNote', () => {
  let wrapper: any;
  let minimalProps;
  let commonProps: any;
  let mockSubmit: any;
  let mockCancel: any;

  beforeEach(() => {
    mockSubmit = jest.fn(() => Promise.resolve({}));
    mockCancel = jest.fn(() => Promise.resolve({}));
    minimalProps = {
      onSubmit: mockSubmit,
      onCancel: mockCancel,
    };
    commonProps = { ...minimalProps, showEditor: true };
    wrapper = mountWithIntl(
      <BrowserRouter>
        <EditableNote {...minimalProps} />
      </BrowserRouter>,
    );
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
    expect(wrapper.find('.note-view-outer-container').length).toBe(1);
  });

  test('test renderActions is called and rendered correctly when showEditor is true', () => {
    wrapper = mountWithIntl(
      <BrowserRouter>
        <EditableNote {...commonProps} />
      </BrowserRouter>,
    );
    expect(wrapper.length).toBe(1);
    expect(wrapper.find('.note-view-outer-container').length).toBe(1);
    expect(wrapper.find('.editable-note-actions').length).toBe(1);
  });

  test('test handleSubmitClick with successful onSubmit', (done) => {
    wrapper.find(EditableNoteImpl).setState({ error: 'should not appear' });
    const instance = wrapper.find(EditableNoteImpl).instance();
    const promise = instance.handleSubmitClick();
    promise.finally(() => {
      expect(mockSubmit).toHaveBeenCalledTimes(1);
      expect(instance.state.error).toEqual(null);
      done();
    });
  });

  test('test handleRenameExperiment errors correctly', (done) => {
    mockSubmit = jest.fn(
      () =>
        new Promise((resolve, reject) => {
          window.setTimeout(() => {
            reject();
          }, 100);
        }),
    );
    minimalProps = {
      onSubmit: mockSubmit,
      onCancel: mockCancel,
    };
    wrapper = mountWithIntl(
      <BrowserRouter>
        <EditableNote {...minimalProps} />
      </BrowserRouter>,
    );

    const instance = wrapper.find(EditableNoteImpl).instance();
    const promise = instance.handleSubmitClick();
    promise.finally(() => {
      wrapper.update();
      expect(mockSubmit).toHaveBeenCalledTimes(1);
      expect(instance.state.error).toEqual('Failed to submit');
      done();
    });
  });
});
