import { Fragment as _Fragment, jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, afterEach, jest, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { Button as AntDButton, ConfigProvider as OrigAntDConfigProvider } from 'antd';
import { DesignSystemProvider, ApplyDesignSystemFlags, DesignSystemAntDConfigProvider, RestoreAntDDefaultClsPrefix, ApplyDesignSystemContextOverrides, } from './DesignSystemProvider';
import { Button } from '../Button';
import { useDesignSystemFlags } from '../Hooks';
import { useDesignSystemContext } from '../Hooks/useDesignSystemContext';
describe('DesignSystemProvider', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    it('can set flags to be consumed by the hooks', () => {
        const TestComponent = () => {
            const flags = useDesignSystemFlags();
            expect(flags.__DEBUG__).toBe(true);
            return _jsx(_Fragment, {});
        };
        render(_jsx(DesignSystemProvider, { flags: {
                __DEBUG__: true,
            }, children: _jsx(TestComponent, {}) }));
        expect.hasAssertions();
    });
    describe('ApplyDesignSystemFlags', () => {
        it('can override flags from the higher provider.', () => {
            const TestComponent = () => {
                const flags = useDesignSystemFlags();
                expect(flags.__DEBUG__).toBe(true);
                return _jsx(_Fragment, {});
            };
            render(_jsx(DesignSystemProvider, { flags: {
                    __DEBUG__: false,
                }, children: _jsx(ApplyDesignSystemFlags, { flags: {
                        __DEBUG__: true,
                    }, children: _jsx(TestComponent, {}) }) }));
            expect.hasAssertions();
        });
        it('does not affect other props of the provider', () => {
            const mockGetPopupContainer = jest.fn();
            const TestComponent = () => {
                const { getPopupContainer } = useDesignSystemContext();
                expect(getPopupContainer).toBe(mockGetPopupContainer);
                return _jsx(_Fragment, {});
            };
            render(_jsx(DesignSystemProvider, { getPopupContainer: mockGetPopupContainer, children: _jsx(ApplyDesignSystemFlags, { flags: {
                        __DEBUG__: true,
                    }, children: _jsx(TestComponent, {}) }) }));
            expect.hasAssertions();
        });
    });
});
describe('ApplyDesignSystemContextOverrides', () => {
    it('can override context from the higher provider.', () => {
        const TestComponent = () => {
            const { getPopupContainer } = useDesignSystemContext();
            expect(getPopupContainer).toBe(mockGetPopupContainer);
            return _jsx(_Fragment, {});
        };
        const mockGetPopupContainer = jest.fn();
        render(_jsx(DesignSystemProvider, { getPopupContainer: mockGetPopupContainer, children: _jsx(ApplyDesignSystemContextOverrides, { getPopupContainer: undefined, children: _jsx(TestComponent, {}) }) }));
        expect.hasAssertions();
        // Validate that `ApplyDesignSystemContextOverrides` changes `getPopupContainer` as expected.
        expect(mockGetPopupContainer).not.toHaveBeenCalled();
    });
    it('does not affect other props of the provider', () => {
        const TestComponent = () => {
            const flags = useDesignSystemFlags();
            const { enableAnimation } = useDesignSystemContext();
            // Validate that `ApplyDesignSystemContextOverrides` changes `flags` as expected.
            expect(flags.__DEBUG__).toBe(false);
            // Validate that prop was not changed.
            expect(enableAnimation).toBe(true);
            return _jsx(_Fragment, {});
        };
        render(_jsx(DesignSystemProvider, { flags: {
                __DEBUG__: true,
            }, enableAnimation: true, children: _jsx(ApplyDesignSystemContextOverrides, { flags: {
                    __DEBUG__: false,
                }, children: _jsx(TestComponent, {}) }) }));
        expect.hasAssertions();
    });
});
describe('AntDConfigProvider', () => {
    const AntDTestComponent = () => {
        return _jsx(AntDButton, { "data-testid": "button", children: "Button" });
    };
    it('renders antd components with default class name when not wrapped with AntDConfigProvider', () => {
        render(_jsx(AntDTestComponent, {}));
        const button = screen.getByTestId('button');
        const className = button.getAttribute('class');
        expect(className).not.toMatch(/du-bois-/);
        expect(className).toMatch(/ant-/);
    });
    it('renders antd components with antd class name even under DesignSystemProvider', () => {
        render(_jsx(DesignSystemProvider, { children: _jsx(AntDTestComponent, {}) }));
        const button = screen.getByTestId('button');
        const className = button.getAttribute('class');
        expect(className).toMatch(/ant-/);
        expect(className).not.toMatch(/du-bois-/);
    });
    it('renders antd components with DS class name wrapped with AntDConfigProvider even with a wrapping overriding antd ConfigProvider', () => {
        const DSTestComponent = () => {
            return (
            // All DS components should also be wrapped with <AntDConfigProvider>,
            // to prevent original AntD ConfigProvider overriding it.
            // This is an example of how a DS component is structured.
            _jsx(DesignSystemAntDConfigProvider, { children: _jsx(AntDTestComponent, {}) }));
        };
        render(_jsx(DesignSystemProvider, { children: _jsx(OrigAntDConfigProvider, { prefixCls: "foobar", children: _jsx(DSTestComponent, {}) }) }));
        const button = screen.getByTestId('button');
        const className = button.getAttribute('class');
        expect(className).toMatch(/du-bois-/);
        expect(className).not.toMatch(/foobar-/);
    });
    it('resets antd class when using RestoreAntDDefaultClsPrefix', () => {
        const DSWrapperComponent = ({ children }) => {
            return (_jsxs(DesignSystemAntDConfigProvider, { children: [_jsx(Button, { componentId: "codegen_design-system_src_design-system_designsystemprovider_designsystemprovider.test.tsx_209", "data-testid": "ds-button", children: "Button" }), _jsx(RestoreAntDDefaultClsPrefix, { children: children })] }));
        };
        render(_jsx(DesignSystemProvider, { children: _jsx(DSWrapperComponent, { children: _jsx(AntDButton, { "data-testid": "ant-button", children: "Button" }) }) }));
        const antButtonClass = screen.getByTestId('ant-button').getAttribute('class');
        const dsButtonClass = screen.getByTestId('ds-button').getAttribute('class');
        expect(antButtonClass).toMatch(/ant-btn/);
        expect(dsButtonClass).toMatch(/du-bois-/);
    });
});
//# sourceMappingURL=DesignSystemProvider.test.js.map