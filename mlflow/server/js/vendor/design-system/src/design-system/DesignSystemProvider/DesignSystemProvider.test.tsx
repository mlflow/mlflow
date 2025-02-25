import { render, screen } from '@testing-library/react';
import { Button as AntDButton, ConfigProvider as OrigAntDConfigProvider } from 'antd';

import {
  DesignSystemProvider,
  ApplyDesignSystemFlags,
  DesignSystemAntDConfigProvider,
  RestoreAntDDefaultClsPrefix,
  ApplyDesignSystemContextOverrides,
} from './DesignSystemProvider';
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
      return <></>;
    };

    render(
      <DesignSystemProvider
        flags={{
          __DEBUG__: true,
        }}
      >
        <TestComponent />
      </DesignSystemProvider>,
    );

    expect.hasAssertions();
  });

  describe('ApplyDesignSystemFlags', () => {
    it('can override flags from the higher provider.', () => {
      const TestComponent = () => {
        const flags = useDesignSystemFlags();
        expect(flags.__DEBUG__).toBe(true);
        return <></>;
      };

      render(
        <DesignSystemProvider
          flags={{
            __DEBUG__: false,
          }}
        >
          <ApplyDesignSystemFlags
            flags={{
              __DEBUG__: true,
            }}
          >
            <TestComponent />
          </ApplyDesignSystemFlags>
        </DesignSystemProvider>,
      );

      expect.hasAssertions();
    });

    it('does not affect other props of the provider', () => {
      const mockGetPopupContainer = jest.fn();

      const TestComponent = () => {
        const { getPopupContainer } = useDesignSystemContext();
        expect(getPopupContainer).toBe(mockGetPopupContainer);
        return <></>;
      };

      render(
        <DesignSystemProvider getPopupContainer={mockGetPopupContainer}>
          <ApplyDesignSystemFlags
            flags={{
              __DEBUG__: true,
            }}
          >
            <TestComponent />
          </ApplyDesignSystemFlags>
        </DesignSystemProvider>,
      );

      expect.hasAssertions();
    });
  });
});

describe('ApplyDesignSystemContextOverrides', () => {
  it('can override context from the higher provider.', () => {
    const TestComponent = () => {
      const { getPopupContainer } = useDesignSystemContext();
      expect(getPopupContainer).toBe(mockGetPopupContainer);
      return <></>;
    };

    const mockGetPopupContainer = jest.fn();

    render(
      <DesignSystemProvider getPopupContainer={mockGetPopupContainer}>
        <ApplyDesignSystemContextOverrides getPopupContainer={undefined}>
          <TestComponent />
        </ApplyDesignSystemContextOverrides>
      </DesignSystemProvider>,
    );

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
      return <></>;
    };

    render(
      <DesignSystemProvider
        flags={{
          __DEBUG__: true,
        }}
        enableAnimation={true}
      >
        <ApplyDesignSystemContextOverrides
          flags={{
            __DEBUG__: false,
          }}
        >
          <TestComponent />
        </ApplyDesignSystemContextOverrides>
      </DesignSystemProvider>,
    );

    expect.hasAssertions();
  });
});

describe('AntDConfigProvider', () => {
  const AntDTestComponent = () => {
    return <AntDButton data-testid="button">Button</AntDButton>;
  };

  it('renders antd components with default class name when not wrapped with AntDConfigProvider', () => {
    render(<AntDTestComponent />);

    const button = screen.getByTestId('button');
    const className = button.getAttribute('class');

    expect(className).not.toMatch(/du-bois-/);
    expect(className).toMatch(/ant-/);
  });

  it('renders antd components with antd class name even under DesignSystemProvider', () => {
    render(
      <DesignSystemProvider>
        <AntDTestComponent />
      </DesignSystemProvider>,
    );

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
        <DesignSystemAntDConfigProvider>
          <AntDTestComponent />
        </DesignSystemAntDConfigProvider>
      );
    };

    render(
      <DesignSystemProvider>
        <OrigAntDConfigProvider prefixCls="foobar">
          <DSTestComponent />
        </OrigAntDConfigProvider>
      </DesignSystemProvider>,
    );

    const button = screen.getByTestId('button');
    const className = button.getAttribute('class');

    expect(className).toMatch(/du-bois-/);
    expect(className).not.toMatch(/foobar-/);
  });

  it('resets antd class when using RestoreAntDDefaultClsPrefix', () => {
    const DSWrapperComponent: React.FC = ({ children }) => {
      return (
        <DesignSystemAntDConfigProvider>
          <Button
            componentId="codegen_design-system_src_design-system_designsystemprovider_designsystemprovider.test.tsx_209"
            data-testid="ds-button"
          >
            Button
          </Button>
          {/* Wrapper components are expected to render children under RestoreAntDDefaultClsPrefix */}
          <RestoreAntDDefaultClsPrefix>{children}</RestoreAntDDefaultClsPrefix>
        </DesignSystemAntDConfigProvider>
      );
    };

    render(
      <DesignSystemProvider>
        <DSWrapperComponent>
          {/* Nesting an ant inside a wrapper component should keep its ant- class prefix */}
          <AntDButton data-testid="ant-button">Button</AntDButton>
        </DSWrapperComponent>
      </DesignSystemProvider>,
    );

    const antButtonClass = screen.getByTestId('ant-button').getAttribute('class');
    const dsButtonClass = screen.getByTestId('ds-button').getAttribute('class');

    expect(antButtonClass).toMatch(/ant-btn/);
    expect(dsButtonClass).toMatch(/du-bois-/);
  });
});
