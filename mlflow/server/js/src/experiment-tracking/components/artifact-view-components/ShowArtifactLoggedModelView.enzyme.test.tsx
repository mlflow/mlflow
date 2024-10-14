/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactLoggedModelView, { ShowArtifactLoggedModelViewImpl } from './ShowArtifactLoggedModelView';
import { mountWithIntl, shallowWithInjectIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

describe('ShowArtifactLoggedModelView', () => {
  let wrapper: any;
  let instance;
  let minimalProps: any;
  let commonProps: any;
  const minimumFlavors = `
flavors:
  python_function:
    loader_module: mlflow.sklearn
`;
  const validMlModelFile =
    minimumFlavors +
    'signature:\n' +
    '  inputs: \'[{"name": "sepal length (cm)", "type": "double"}, {"name": "sepal width\n' +
    '    (cm)", "type": "double"}, {"name": "petal length (cm)", "type": "double"}, {"name":\n' +
    '    "petal width (cm)", "type": "double"}]\'\n' +
    '  outputs: \'[{"type": "long"}]\'';

  beforeEach(() => {
    minimalProps = { path: 'fakePath', runUuid: 'fakeUuid', artifactRootUri: 'fakeRootUri' };
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(minimumFlavors);
    });
    commonProps = { ...minimalProps, getArtifact };
    wrapper = shallowWithInjectIntl(<ShowArtifactLoggedModelView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
  });

  test('should render error message when error occurs', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error('my error text'));
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = shallowWithInjectIntl(<ShowArtifactLoggedModelView {...props} />);
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
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
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
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(0);
      done();
    });
  });

  test('should not break schema table when inputs only in MLmodel file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(
        minimumFlavors +
          'signature:\n' +
          '  inputs: \'[{"name": "sepal length (cm)", "type": "double"}, {"name": "sepal width\n' +
          '    (cm)", "type": "double"}, {"name": "petal length (cm)", "type": "double"}, {"name":\n' +
          '    "petal width (cm)", "type": "double"}]\'\n',
      );
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(1);
      done();
    });
  });

  test('should not break schema table when outputs only in MLmodel file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(
        minimumFlavors +
          'signature:\n' +
          '  outputs: \'[{"name": "sepal length (cm)", "type": "double"}, {"name": "sepal width\n' +
          '    (cm)", "type": "double"}, {"name": "petal length (cm)", "type": "double"}, {"name":\n' +
          '    "petal width (cm)", "type": "double"}]\'\n',
      );
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(1);
      done();
    });
  });

  test('should not break schema table when no inputs or outputs in MLmodel file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(minimumFlavors + 'signature:');
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(1);
      done();
    });
  });

  test('should render code group and code snippet', (done) => {
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-code-group').length).toBe(1);
      expect(wrapper.find('.artifact-logged-model-view-code-content').length).toBe(2);
      done();
    });
  });

  test('should find model path in code snippet', (done) => {
    const props = { ...commonProps, path: 'modelPath', artifactRootUri: 'some/root' };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-code-content').at(1).html()).toContain(
        'runs:/fakeUuid/modelPath',
      );
      done();
    });
  });

  test('should render serving input validation in code snippet', (done) => {
    const props = { ...commonProps, path: 'modelPath', artifactRootUri: 'some/root' };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-code-content').at(0).text()).toContain(
        'validate_serving_input(model_uri, serving_payload)',
      );
      done();
    });
  });

  test('should suggest registration when model not registered', (done) => {
    const props = { ...commonProps };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-header').text()).toContain('You can also');
      done();
    });
  });

  test('should not suggest registration when model already registered', (done) => {
    const props = { ...commonProps, registeredModelLink: 'someLink' };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-header').text()).toContain(
        'This model is also registered to the',
      );
      done();
    });
  });

  test('should fetch artifacts and serving input on component update', () => {
    instance = wrapper.instance();
    instance.fetchLoggedModelMetadata = jest.fn();
    instance.fetchServingInputExample = jest.fn();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchLoggedModelMetadata).toBeCalled();
    expect(instance.props.getArtifact).toBeCalled();
  });

  test('should render code snippet with original flavor when no pyfunc flavor', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(`
flavors:
  sklearn:
    version: 1.2.3
`);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
      expect(impl.state().flavor).toBe('sklearn');
      const codeContent = impl.find('.artifact-logged-model-view-code-content');
      expect(codeContent.length).toBe(2);
      expect(codeContent.at(1).text().includes('mlflow.sklearn.load_model')).toBe(true);
      done();
    });
  });

  test('should not render code snippet for mleap flavor', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(`
flavors:
  mleap:
    version: 1.2.3
`);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
      expect(impl.state().flavor).toBe('mleap');
      // Only validate model serving code snippet is rendered
      expect(impl.find('.artifact-logged-model-view-code-content').length).toBe(1);
      done();
    });
  });

  test('should render serving validation code snippet if serving_input_example exists', (done) => {
    const getArtifact = jest
      .fn()
      .mockImplementationOnce((artifactLocation) => {
        return Promise.resolve(`
flavors:
  sklearn:
    version: 1.2.3
saved_input_example_info:
  serving_input_path: serving_input_example.json
`);
      })
      .mockImplementationOnce((artifactLocation) => {
        return Promise.resolve(`
{
  "dataframe_split": {
    "columns": [
      "messages"
    ],
    "data": [
      [
        [
          {
            "role": "user",
            "content": "some question"
          }
        ]
      ]
    ]
  }
}
`);
      });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
      expect(impl.state().serving_input).toBeDefined();
      const codeContent = impl.find('.artifact-logged-model-view-code-content');
      expect(codeContent.length).toBe(2);
      const codeContentText = codeContent.at(0).text();
      expect(codeContentText.includes('# The model is logged with an input example')).toBe(true);
      expect(codeContentText.includes('validate_serving_input(model_uri, serving_payload)')).toBe(true);
      done();
    });
  });

  test('should render serving validation code snippet if serving_input_example does not exist', (done) => {
    const getArtifact = jest
      .fn()
      .mockImplementationOnce((artifactLocation) => {
        return Promise.resolve(`
flavors:
  sklearn:
    version: 1.2.3
`);
      })
      .mockImplementationOnce((artifactLocation) => {
        return Promise.reject(new Error('file not existing'));
      });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactLoggedModelView {...props} />);
    setImmediate(() => {
      wrapper.update();
      const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
      expect(impl.state().serving_input).toBeDefined();
      const codeContent = impl.find('.artifact-logged-model-view-code-content');
      expect(codeContent.length).toBe(2);
      const codeContentText = codeContent.at(0).text();
      expect(codeContentText.includes('# The logged model does not contain an input_example')).toBe(true);
      expect(codeContentText.includes('validate_serving_input(model_uri, serving_payload)')).toBe(true);
      done();
    });
  });
});
