import { useEffect, useMemo, useState } from 'react';
import ReactDOM from 'react-dom';
import { CacheProvider, Global } from '@emotion/react';
import createCache from '@emotion/cache';
import { prefixer } from 'stylis';
import { stylisExtraScopePlugin } from './stylisExtraScopePlugin';
import { MFEAttributesContextProvider } from './MFEAttributesContext';
import type { MFECustomActionCallbacks } from './MFEAttributesContext';
import { MLFlowMFERoot } from './MLFlowMFERoot';

/**
 * A component with a minimal set of fundamental styles: container, font-family, reference font-size etc.
 */
const GlobalMlflowStyles = () => (
  <Global
    styles={{
      '.mlflow-wc-root': {
        height: '100%',
        fontFamily: `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial,
          'Noto Sans', sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
          'Noto Color Emoji', 'FontAwesome'`,
        fontSize: 13,
        lineHeight: '18px',
        fontWeight: 400,
        boxShadow: 'none',
      },
    }}
  />
);

/**
 * Wrapper for MLflow root that listens to overarching HTMLElement's mlflow-specific
 * callback function updates and reinjects it downstream as a attribute object.
 */
const MLflowWebComponent = ({
  onCustomCallbacksChange,
}: {
  onCustomCallbacksChange: (
    callbackFn: (newCallbackList: MFECustomActionCallbacks) => void,
  ) => () => void;
}) => {
  // Store actually registered callbacks in the stateful value
  const [registeredCallbacks, setRegisteredCallbacks] = useState<MFECustomActionCallbacks>({});

  useEffect(() => {
    // Upon component mount, register a listener that will fire up on every change
    // to the custom callback list
    const unregisterListener = onCustomCallbacksChange((callbacks: MFECustomActionCallbacks) => {
      setRegisteredCallbacks(callbacks);
    });

    return () => unregisterListener();
  }, [onCustomCallbacksChange]);

  // Assemble the attributes object
  const attributes = useMemo(
    () => ({ customActionCallbacks: registeredCallbacks }),
    [registeredCallbacks],
  );

  // Render MLflow application
  return <MLFlowMFERoot attributes={attributes} />;
};

const registerMlflowWebComponent = () => {
  window.customElements.define(
    'mlflow-ui',
    class MLflowUIWebComponent extends HTMLElement {
      private static _styles: HTMLStyleElement[] = [];
      private static _instances: MLflowUIWebComponent[] = [];
      private _shadowRoot: ShadowRoot;
      private _container: HTMLDivElement;
      private _customCallbacks: MFECustomActionCallbacks = {};
      private _customCallbacksChangeListeners: ((
        newCallbackList: MFECustomActionCallbacks,
      ) => void)[] = [];

      constructor() {
        super();

        // Create a application mount point within Shadow DOM
        this._container = document.createElement('DIV') as HTMLDivElement;
        this._container.classList.add('mlflow-wc-root');

        this._shadowRoot = this.attachShadow({ mode: 'open' });
        this._shadowRoot.appendChild(this._container);

        // Register an instance within static set
        MLflowUIWebComponent._instances.push(this);

        // If there are any styles already existing, reinject them here
        MLflowUIWebComponent._styles.forEach((existingStyle) => {
          this._shadowRoot.appendChild(existingStyle.cloneNode(true));
        });

        // Each <mlflow-ui> element will have "addMlflowListener" method
        // used to register custom callbacks, e.g. for intercepting model registration.
        Object.defineProperty(this, 'addMlflowListener', {
          value: <T extends keyof MFECustomActionCallbacks>(
            key: T,
            callback: MFECustomActionCallbacks[T],
          ) => {
            this._customCallbacks[key] = callback;
            this._customCallbacksChangeListeners.forEach((listener) =>
              listener(this._customCallbacks),
            );
          },
          configurable: false,
        });
      }

      // Injects a new style element to all web component instances
      static webpackInjectStyle(element: HTMLStyleElement) {
        MLflowUIWebComponent._styles.push(element);
        MLflowUIWebComponent._instances.forEach((existingInstance) => {
          existingInstance._shadowRoot.appendChild(element.cloneNode(true));
        });
      }

      connectedCallback() {
        const isSafariVersion15 = Boolean(
          navigator.vendor.match(/apple/i) && navigator.userAgent.match(/version\/15\.0/i),
        );

        // Create a listener that will react to updating custom callback registry
        const registerCustomCallbackListChange = (
          callbackFn: (newCallbackList: MFECustomActionCallbacks) => void,
        ) => {
          this._customCallbacksChangeListeners.push(callbackFn);
          callbackFn(this._customCallbacks);

          return () => {
            this._customCallbacksChangeListeners = this._customCallbacksChangeListeners.filter(
              (existingCallback) => existingCallback !== callbackFn,
            );
          };
        };

        const emotionCache = createCache({
          key: 'mlflow-css',
          // We need to inject styles directly into Shadow DOM root
          // @ts-expect-error emotion types are not valid with
          container: this._shadowRoot,
          // @ts-expect-error emotion types are not compatible with prefixes
          stylisPlugins: [prefixer, stylisExtraScopePlugin('.mlflow-wc-root')],
          // Fixes a bug with emotion and shadow dom in Safari
          speedy: !isSafariVersion15,
        });

        ReactDOM.render(
          <MFEAttributesContextProvider value={{}}>
            <CacheProvider value={emotionCache}>
              <MLflowWebComponent onCustomCallbacksChange={registerCustomCallbackListChange} />
              <GlobalMlflowStyles />
            </CacheProvider>
          </MFEAttributesContextProvider>,
          this._container,
        );
      }
    },
  );
};

registerMlflowWebComponent();
