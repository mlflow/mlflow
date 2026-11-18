import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { useContext } from 'react';

import { ModelTraceExplorerAttributesTab } from './ModelTraceExplorerAttributesTab';
import { ModelTraceExplorerPayloadCacheProvider } from './ModelTraceExplorerPayloadCache';
import { ModelTraceExplorerPreferencesProvider } from '../ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerViewStateContext } from '../ModelTraceExplorerViewStateContext';
import { MOCK_CHAT_SPAN } from '../../ModelTraceExplorer.test-utils';
import type { ModelTraceSpanNode } from '../ModelTrace.types';

describe('attribute payload reuse', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it.each(['toString', 'constructor', '__proto__'])(
    'preserves a comparison attribute named %s missing on the other side',
    (key) => {
      const attributes = Object.fromEntries([[key, 'own attribute value']]);
      const Comparison = ({ children }: { children: React.ReactNode }) => {
        const state = useContext(ModelTraceExplorerViewStateContext);
        return (
          <ModelTraceExplorerViewStateContext.Provider
            value={{
              ...state,
              selectedComparisonNode: { ...MOCK_CHAT_SPAN, attributes: {} },
              selectedNodeComparisonRole: 'baseline',
            }}
          >
            {children}
          </ModelTraceExplorerViewStateContext.Provider>
        );
      };
      const { container } = render(
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <Comparison>
              <ModelTraceExplorerAttributesTab
                activeSpan={{ ...MOCK_CHAT_SPAN, attributes }}
                searchFilter=""
                activeMatch={null}
              />
            </Comparison>
          </DesignSystemProvider>
        </IntlProvider>,
      );
      expect(container).toHaveTextContent(key);
      expect(container).toHaveTextContent('own attribute value');
    },
  );

  it('retains attribute JSON across rerenders, remounts and format changes, then replaces changed payloads', async () => {
    const attributes = { payload: { body: 'attribute café 日本語 🙂' }, empty: null, enabled: false, count: 0 };
    const span: ModelTraceSpanNode = { ...MOCK_CHAT_SPAN, attributes };
    const stringify = jest.spyOn(JSON, 'stringify');
    const view = (show: boolean, activeSpan = span) => (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <ModelTraceExplorerPreferencesProvider initialRenderMode="json">
            <ModelTraceExplorerPayloadCacheProvider>
              {show && <ModelTraceExplorerAttributesTab activeSpan={activeSpan} searchFilter="" activeMatch={null} />}
            </ModelTraceExplorerPayloadCacheProvider>
          </ModelTraceExplorerPreferencesProvider>
        </DesignSystemProvider>
      </IntlProvider>
    );
    const { rerender, container } = render(view(true));
    rerender(view(true, { ...span }));
    rerender(view(false));
    rerender(view(true));
    expect(stringify.mock.calls.filter(([value]) => value === attributes)).toHaveLength(1);
    expect(screen.getByText('"attribute café 日本語 🙂"')).toBeInTheDocument();
    const replacement = { ...attributes, payload: { body: 'replacement attribute' } };
    rerender(view(true, { ...span, attributes: replacement }));
    expect(stringify.mock.calls.filter(([value]) => value === replacement)).toHaveLength(1);
    expect(screen.getByText('"replacement attribute"')).toBeInTheDocument();
    const parse = jest.spyOn(JSON, 'parse');
    await userEvent.click(screen.getByRole('button', { name: 'JSON' }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Pretty' }));
    rerender(view(true, { ...span, attributes: replacement }));
    expect(stringify.mock.calls.filter(([value]) => value === replacement)).toHaveLength(1);
    expect(stringify.mock.calls.filter(([value]) => value === replacement.payload)).toHaveLength(1);
    expect(parse.mock.calls.filter(([value]) => String(value).includes('replacement attribute'))).toHaveLength(0);
    expect(container).toHaveTextContent('replacement attribute');
  });
});
