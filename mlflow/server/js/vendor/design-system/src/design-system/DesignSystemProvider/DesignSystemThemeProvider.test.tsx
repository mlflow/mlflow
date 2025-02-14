import { render } from '@testing-library/react';

import { DesignSystemThemeProvider, DesignSystemProvider } from './DesignSystemProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';

describe('DesignSystemThemeProvider', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it.each([{ value: true }, { value: false }])('sets the dark mode value to $value in the context', ({ value }) => {
    const TestComponent = () => {
      const { theme } = useDesignSystemTheme();
      expect(theme.isDarkMode).toBe(value);
      return <></>;
    };

    render(
      // eslint-disable-next-line react/forbid-elements
      <DesignSystemThemeProvider isDarkMode={value}>
        <DesignSystemProvider>
          <TestComponent />
        </DesignSystemProvider>
      </DesignSystemThemeProvider>,
    );
  });
});
