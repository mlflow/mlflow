import React from 'react';
import { render, screen } from '../utils/TestUtils.react18';
import { DesignSystemContainer } from './DesignSystemContainer';
import { message } from 'antd';

let mockGetPopupContainerFn: any;

jest.mock('antd', () => ({
  ...jest.requireActual('antd'),
  message: { config: jest.fn() },
  ConfigProvider: ({ children }: any) => children,
}));

jest.mock('@databricks/design-system', () => ({
  DesignSystemProvider: ({ getPopupContainer, children }: any) => {
    mockGetPopupContainerFn = getPopupContainer;
    return children;
  },
  DesignSystemThemeProvider: ({ children }: any) => {
    return children;
  },
}));

describe('DesignSystemContainer', () => {
  window.customElements.define(
    'demo-shadow-dom',
    class extends HTMLElement {
      _shadowRoot: any;
      constructor() {
        super();
        this._shadowRoot = this.attachShadow({ mode: 'open' });
      }
      connectedCallback() {
        render(
          <DesignSystemContainer>
            <span>hello in shadow dom</span>
          </DesignSystemContainer>,
          {
            baseElement: this._shadowRoot,
          },
        );
      }
    },
  );

  test('should not attach additional container while in document.body', () => {
    render(
      <DesignSystemContainer>
        <span>hello</span>
      </DesignSystemContainer>,
    );
    expect(message.config).toBeCalledTimes(0);
    expect(screen.getByText('hello')).toBeInTheDocument();
    expect(mockGetPopupContainerFn()).toBe(document.body);
  });

  test('should attach additional container while in shadow DOM', () => {
    const customElement = window.document.createElement('demo-shadow-dom');
    window.document.body.appendChild(customElement);

    expect(message.config).toBeCalledTimes(1);
    expect(mockGetPopupContainerFn()).not.toBe(document.body);
    expect(mockGetPopupContainerFn().tagName).toBe('DIV');

    expect(1).toBe(1);
  });
});
