import React from 'react';
import { mount } from 'enzyme';
import { DesignSystemContainer } from './DesignSystemContainer';
import { message } from 'antd';

let mockGetPopupContainerFn;

jest.mock('antd');
jest.mock('@databricks/design-system', () => ({
  DesignSystemProvider: ({ getPopupContainer, children }) => {
    mockGetPopupContainerFn = getPopupContainer;
    return children;
  },
}));

describe('DesignSystemContainer', () => {
  window.customElements.define(
    'demo-shadow-dom',
    class extends HTMLElement {
      constructor() {
        super();
        this._shadowRoot = this.attachShadow({ mode: 'open' });
      }
      connectedCallback() {
        mount(
          <DesignSystemContainer>
            <span>hello in shadow dom</span>
          </DesignSystemContainer>,
          {
            attachTo: this._shadowRoot,
          },
        );
      }
    },
  );

  test('should not attach additional container while in document.body', () => {
    const wrapper = mount(
      <body>
        <DesignSystemContainer>
          <span>hello</span>
        </DesignSystemContainer>
      </body>,
      { attachTo: document.documentElement },
    );
    expect(message.config).toBeCalledTimes(0);
    expect(wrapper.length).toBe(1);
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
