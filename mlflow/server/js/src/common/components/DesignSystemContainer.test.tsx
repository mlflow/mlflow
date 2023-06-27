/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { mount } from 'enzyme';
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
}));

jest.mock('@databricks/web-shared/design-system', () => ({
  SupportsDuBoisThemes: ({ children }: any) => {
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
