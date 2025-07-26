import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { RunNameHeaderCellRendererErrorBoundary } from './RunNameHeaderCellRendererErrorBoundary';

// Mock design system theme
jest.mock('@databricks/design-system', () => ({
  Button: ({ children, onClick, ...props }: any) => (
    <button onClick={onClick} {...props}>
      {children}
    </button>
  ),
  useDesignSystemTheme: () => ({
    theme: {
      spacing: { xs: 4, sm: 8 },
      colors: { textSecondary: '#666' },
      typography: { fontSizeXs: '12px' },
    },
  }),
}));

const ThrowError = ({ shouldThrow }: { shouldThrow: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>Working component</div>;
};

describe('RunNameHeaderCellRendererErrorBoundary', () => {
  beforeEach(() => {
    // Suppress console.error for these tests to avoid noise
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const renderWithIntl = (component: React.ReactElement) => {
    return render(<IntlProvider locale="en">{component}</IntlProvider>);
  };

  it('renders children when there is no error', () => {
    renderWithIntl(
      <RunNameHeaderCellRendererErrorBoundary>
        <ThrowError shouldThrow={false} />
      </RunNameHeaderCellRendererErrorBoundary>,
    );

    expect(screen.getByText('Working component')).toBeInTheDocument();
  });

  it('renders error fallback when child component throws', () => {
    renderWithIntl(
      <RunNameHeaderCellRendererErrorBoundary>
        <ThrowError shouldThrow />
      </RunNameHeaderCellRendererErrorBoundary>,
    );

    expect(screen.getByText('Run Name')).toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('renders custom fallback component when provided', () => {
    const customFallback = <div>Custom fallback</div>;

    renderWithIntl(
      <RunNameHeaderCellRendererErrorBoundary fallbackComponent={customFallback}>
        <ThrowError shouldThrow />
      </RunNameHeaderCellRendererErrorBoundary>,
    );

    expect(screen.getByText('Custom fallback')).toBeInTheDocument();
    expect(screen.queryByText('Run Name')).not.toBeInTheDocument();
  });

  it('retry button is clickable when error occurs', () => {
    renderWithIntl(
      <RunNameHeaderCellRendererErrorBoundary>
        <ThrowError shouldThrow />
      </RunNameHeaderCellRendererErrorBoundary>,
    );

    // Error state should be shown
    expect(screen.getByText('Run Name')).toBeInTheDocument();

    const retryButton = screen.getByText('Retry');
    expect(retryButton).toBeInTheDocument();

    // Click retry button should not throw an error
    expect(() => {
      fireEvent.click(retryButton);
    }).not.toThrow();
  });

  it('logs error to console when component catches error', () => {
    const consoleSpy = jest.spyOn(console, 'error');

    renderWithIntl(
      <RunNameHeaderCellRendererErrorBoundary>
        <ThrowError shouldThrow />
      </RunNameHeaderCellRendererErrorBoundary>,
    );

    expect(consoleSpy).toHaveBeenCalledWith('RunNameHeaderCellRenderer error:', expect.any(Error), expect.any(Object));
  });
});
