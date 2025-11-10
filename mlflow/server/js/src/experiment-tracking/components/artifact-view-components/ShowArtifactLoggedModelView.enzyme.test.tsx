/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { describe, beforeEach, jest, test, expect } from '@jest/globals';
import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactLoggedModelView, { ShowArtifactLoggedModelViewImpl } from './ShowArtifactLoggedModelView';
import { mountWithIntl, shallowWithInjectIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { DesignSystemProvider } from '@databricks/design-system';

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
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...commonProps} />
      </DesignSystemProvider>,
    );
  });

  test('should render with minimal props without exploding', () => {
    expect(wrapper.length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render error message when error occurs', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error('my error text'));
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );

    // Wait for the promise rejection to be processed
    setTimeout(() => {
      wrapper.update();
      const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
      expect(wrapper.find('.artifact-logged-model-view-error').length).toBeGreaterThan(0);
      expect(impl.state().loading).toBe(false);
      expect(impl.state().error).toBeDefined();
      done();
    }, 100);
  });

  test('should render loading text when view is loading', () => {
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...commonProps} />
      </DesignSystemProvider>,
    );
    const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
    impl.instance().setState({ loading: true });
    wrapper.update();
    expect(wrapper.find('.artifact-logged-model-view-loading').length).toBeGreaterThan(0);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render schema table when valid signature in MLmodel file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(validMlModelFile);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(1);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should not render schema table when invalid signature in MLmodel file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(validMlModelFile + '\nhahaha');
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(0);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
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
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(1);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
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
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(1);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should not break schema table when no inputs or outputs in MLmodel file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(minimumFlavors + 'signature:');
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-schema-table').length).toBe(1);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render code group and code snippet', (done) => {
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...commonProps} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-code-group').length).toBe(1);
      expect(wrapper.find('.artifact-logged-model-view-code-content').length).toBe(2);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should find model path in code snippet', (done) => {
    const props = { ...commonProps, path: 'modelPath', artifactRootUri: 'some/root' };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-code-content').at(1).html()).toContain(
        'runs:/fakeUuid/modelPath',
      );
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render models predict in code snippet', (done) => {
    const props = { ...commonProps, path: 'modelPath', artifactRootUri: 'some/root' };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-code-content').at(0).text()).toContain('mlflow.models.predict');
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should suggest registration when model not registered', (done) => {
    const props = { ...commonProps };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-header').text()).toContain('You can also');
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should not suggest registration when model already registered', (done) => {
    const props = { ...commonProps, registeredModelLink: 'someLink' };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.artifact-logged-model-view-header').text()).toContain(
        'This model is also registered to the',
      );
      done();
    });
  });

  test('should fetch artifacts and serving input on component update', () => {
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...commonProps} />
      </DesignSystemProvider>,
    );
    const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
    const instance = impl.instance();
    instance.fetchLoggedModelMetadata = jest.fn();
    wrapper.setProps({ children: <ShowArtifactLoggedModelView {...commonProps} path="newpath" runUuid="newRunId" /> });
    wrapper.update();
    expect(instance.fetchLoggedModelMetadata).toHaveBeenCalled();
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render code snippet with original flavor when no pyfunc flavor', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(`
flavors:
  sklearn:
    version: 1.2.3
`);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
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

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should not render code snippet for mleap flavor', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(`
flavors:
  mleap:
    version: 1.2.3
`);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
      expect(impl.state().flavor).toBe('mleap');
      // Only validate model serving code snippet is rendered
      expect(impl.find('.artifact-logged-model-view-code-content').length).toBe(1);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render serving validation code snippet if serving_input_example exists', (done) => {
    const getArtifact = jest.fn().mockImplementationOnce((artifactLocation) => {
      return Promise.resolve(`
flavors:
  python_function:
    python_version: 3.9.18
saved_input_example_info:
  artifact_path: input_example.json
`);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
      expect(impl.state().hasInputExample).toBe(true);
      const codeContent = impl.find('.artifact-logged-model-view-code-content');
      expect(codeContent.length).toBe(2);
      const codeContentText = codeContent.at(0).text();
      expect(codeContentText.includes('input_data = pyfunc_model.input_example')).toBe(true);
      expect(codeContentText.includes('mlflow.models.predict')).toBe(true);
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render serving validation code snippet if serving_input_example does not exist', (done) => {
    const getArtifact = jest.fn().mockImplementationOnce((artifactLocation) => {
      return Promise.resolve(`
flavors:
  sklearn:
    version: 1.2.3
`);
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(
      <DesignSystemProvider>
        <ShowArtifactLoggedModelView {...props} />
      </DesignSystemProvider>,
    );
    setImmediate(() => {
      wrapper.update();
      const impl = wrapper.find(ShowArtifactLoggedModelViewImpl);
      expect(impl.state().hasInputExample).toBe(false);
      const codeContent = impl.find('.artifact-logged-model-view-code-content');
      expect(codeContent.length).toBe(2);
      const codeContentText = codeContent.at(0).text();
      expect(codeContentText.includes('# Replace INPUT_EXAMPLE with your own input example to the model')).toBe(true);
      expect(codeContentText.includes('mlflow.models.predict')).toBe(true);
      done();
    });
  });
});
