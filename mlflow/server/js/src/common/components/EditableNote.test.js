import React from 'react';
import { shallow } from 'enzyme';
import { EditableNote } from './EditableNote';

describe('EditableNote', () => {
  let wrapper;
  let minimalProps;
  let commonProps;
  let mockSubmit;
  let mockCancel;

  beforeEach(() => {
    mockSubmit = jest.fn(() => Promise.resolve({}));
    mockCancel = jest.fn(() => Promise.resolve({}));
    minimalProps = {
      onSubmit: mockSubmit,
      onCancel: mockCancel,
    };
    commonProps = { ...minimalProps, showEditor: true };
    wrapper = shallow(<EditableNote {...minimalProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
    expect(wrapper.find('.note-view-outer-container').length).toBe(1);
  });

  test('test renderActions is called and rendered correctly when showEditor is true', () => {
    wrapper = shallow(<EditableNote {...commonProps} />);
    expect(wrapper.length).toBe(1);
    expect(wrapper.find('.note-view-outer-container').length).toBe(1);
    expect(wrapper.find('.editable-note-actions').length).toBe(1);
  });

  test('test handleSubmitClick with successful onSubmit', (done) => {
    wrapper.setState({ error: 'should not appear' });
    const instance = wrapper.instance();
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
    wrapper = shallow(<EditableNote {...minimalProps} />);

    const instance = wrapper.instance();
    const promise = instance.handleSubmitClick();
    promise.finally(() => {
      wrapper.update();
      expect(mockSubmit).toHaveBeenCalledTimes(1);
      expect(instance.state.error).toEqual('Failed to submit');
      done();
    });
  });
});
