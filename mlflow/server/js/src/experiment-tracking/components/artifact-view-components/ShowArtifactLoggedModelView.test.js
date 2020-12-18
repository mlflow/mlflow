import React from 'react';
import { shallow, mount } from 'enzyme';
import ShowArtifactLoggedModelView from './ShowArtifactLoggedModelView';

describe('ShowArtifactLoggedModelView', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let commonProps;
  const validMlModelFile =
    'signature:\n' +
    '  inputs: \'[{"name": "sepal length (cm)", "type": "double"}, {"name": "sepal width\n' +
    '    (cm)", "type": "double"}, {"name": "petal length (cm)", "type": "double"}, {"name":\n' +
    '    "petal width (cm)", "type": "double"}]\'\n' +
    '  outputs: \'[{"type": "long"}]\'';

  beforeEach(() => {
    minimalProps = { path: 'fakePath', runUuid: 'fakeUuid', artifactRootUri: 'fakeRootUri' };
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('some content');
    });
    commonProps = { ...minimalProps, getArtifact };
    wrapper = shallow(<ShowArtifactLoggedModelView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactLoggedModelView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render error message when error occurs', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error('my error text'));
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = shallow(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      expect(wrapper.find('.artifact-logged-model-view-error').length).toBe(1);
      expect(wrapper.instance().state.loading).toBe(false);
      expect(wrapper.instance().state.error).toBeDefined();
      done();
    });
  });

  test('should render loading text when view is loading', () => {
    instance = wrapper.instance();
    instance.setState({ loading: true });
    expect(wrapper.find('.artifact-logged-model-view-loading').length).toBe(1);
  });

  test('should render schema table when valid signature in MLmodel file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(validMlModelFile);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mount(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(1);
      done();
    });
  });

  test('should not render schema table when invalid signature in MLmodel file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(validMlModelFile + '\nhahaha');
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mount(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(0);
      done();
    });
  });

  test('should render code group and code snippet', (done) => {
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-code-group').length).toBe(1);
      expect(wrapper.find('.artifact-logged-model-view-code-content').length).toBe(1);
      done();
    });
  });

  test('should find model path in code snippet', (done) => {
    const props = { ...commonProps, path: 'modelPath', artifactRootUri: 'some/root' };
    wrapper = mount(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-code-content').html()).toContain(
        'some/root/modelPath',
      );
      done();
    });
  });

  test('should suggest registration when model not registered', (done) => {
    const props = { ...commonProps };
    wrapper = mount(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-header').html()).toContain('You can also');
      done();
    });
  });

  test('should not suggest registration when model already registered', (done) => {
    const props = { ...commonProps, registeredModelLink: 'someLink' };
    wrapper = mount(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-header').html()).toContain(
        'This model is also registered to the',
      );
      done();
    });
  });

  test('should fetch artifacts on component update', () => {
    instance = wrapper.instance();
    instance.fetchLoggedModelMetadata = jest.fn();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchLoggedModelMetadata).toBeCalled();
    expect(instance.props.getArtifact).toBeCalled();
  });
});
