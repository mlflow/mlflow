import { Global } from '@emotion/react';
import { act } from 'react-dom/test-utils';
import './MLFlowWebComponent';
import { MLFlowRoot } from '../app';
import { useMFEAttributes } from './MFEAttributesContext';

// Mock the actual MLflow UI root, we don't want to process
// entire app while compiling this test
jest.mock('../app', () => ({
  MLFlowRoot: jest.fn().mockImplementation(() => <div />),
}));

// Mock emotion's <Global /> style component, we don't need to
// deal with real styles in the test
jest.mock('@emotion/react', () => ({
  ...jest.requireActual('@emotion/react'),
  Global: jest.fn().mockImplementation(() => <div />),
}));

const MLFlowRootComponentMock = MLFlowRoot as jest.Mock;

const mountWebComponent = () => {
  const customElement = window.document.createElement('mlflow-ui');
  window.document.body.appendChild(customElement);

  const { shadowRoot } = customElement;
  return {
    shadowRoot: shadowRoot as ShadowRoot,
    // eslint-disable-next-line @typescript-eslint/ban-types
    customElement: customElement as HTMLElement & { addMlflowListener: Function },
  };
};

describe('MLFlowWebComponent', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });
  afterEach(() => {
    window.document.querySelectorAll('mlflow-ui').forEach((e) => e.remove());
  });
  it('should render with mocked application component included', () => {
    // Mock the MLflow UI root to render some test data
    MLFlowRootComponentMock.mockImplementation(() => <div>test MLFlow UI</div>);
    const { shadowRoot } = mountWebComponent();

    // Assert that the properly scoped global styles are mounted
    expect(Global).toBeCalledWith(
      expect.objectContaining({
        styles: expect.objectContaining({ '.mlflow-wc-root': expect.anything() }),
      }),
      expect.anything(),
    );

    // Expect resulting web component to include the mocked MLflow application
    expect(shadowRoot.innerHTML).toContain('test MLFlow UI');
  });

  const mockedRegularRegisterModelFn = jest.fn();
  const mockOnListenerRegisterModelFn = jest.fn();

  // Mock the MLflow UI root to render a button that will check if
  // there is overriden callback provided and if true, call it instead
  // of the regular version
  const MockRegisterModelCallbackComponent = () => {
    const attrs = useMFEAttributes();
    return (
      <button
        onClick={() => {
          const { registerModel } = attrs.customActionCallbacks || {};
          if (registerModel) {
            registerModel({
              run_uuid: 12345,
            });
          } else {
            mockedRegularRegisterModelFn({
              run_uuid: 12345,
            });
          }
        }}
      ></button>
    );
  };

  it('should render and properly call custom callbacks when provided', () => {
    MLFlowRootComponentMock.mockImplementation(() => <MockRegisterModelCallbackComponent />);
    const { customElement, shadowRoot } = mountWebComponent();
    const registerModelButton = shadowRoot.querySelector('button') as HTMLButtonElement;

    act(() => {
      customElement.addMlflowListener('registerModel', mockOnListenerRegisterModelFn);
    });

    registerModelButton.click();
    expect(mockedRegularRegisterModelFn).toBeCalledTimes(0);
    expect(mockOnListenerRegisterModelFn).toBeCalledWith({
      run_uuid: 12345,
    });
  });

  it('should render and call regular functions if custom callbacks not provided', () => {
    MLFlowRootComponentMock.mockImplementation(() => <MockRegisterModelCallbackComponent />);
    const { shadowRoot } = mountWebComponent();
    const registerModelButton = shadowRoot.querySelector('button') as HTMLButtonElement;

    registerModelButton.click();

    expect(mockOnListenerRegisterModelFn).toBeCalledTimes(0);
    expect(mockedRegularRegisterModelFn).toBeCalledWith({
      run_uuid: 12345,
    });
  });
});
