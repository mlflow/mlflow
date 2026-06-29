import { describe, it, expect } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { MCPServerIcon } from './MCPServerIcon';

const renderIcon = (props: React.ComponentProps<typeof MCPServerIcon>) =>
  render(
    <DesignSystemProvider>
      <MCPServerIcon {...props} />
    </DesignSystemProvider>,
  );

const getImg = (container: HTMLElement) => container.querySelector('img');
const getSvg = (container: HTMLElement) => container.querySelector('svg');

describe('MCPServerIcon', () => {
  it('renders fallback McpIcon when no icons provided', () => {
    const { container } = renderIcon({});
    expect(getImg(container)).toBeNull();
    expect(getSvg(container)).toBeTruthy();
  });

  it('renders img when icon src is provided', () => {
    const { container } = renderIcon({ icons: [{ src: 'https://example.com/icon.svg' }] });
    expect(getImg(container)).toHaveAttribute('src', 'https://example.com/icon.svg');
  });

  it('uses server name as alt text', () => {
    renderIcon({ icons: [{ src: 'https://example.com/icon.svg' }], name: 'My Server' });
    expect(screen.getByRole('img')).toHaveAttribute('alt', 'My Server');
  });

  it('uses empty alt when name is not provided', () => {
    const { container } = renderIcon({ icons: [{ src: 'https://example.com/icon.svg' }] });
    expect(getImg(container)).toHaveAttribute('alt', '');
  });

  it('falls back to McpIcon when img fails to load', () => {
    const { container } = renderIcon({ icons: [{ src: 'https://example.com/broken.svg' }] });
    fireEvent.error(getImg(container)!);
    expect(getImg(container)).toBeNull();
    expect(getSvg(container)).toBeTruthy();
  });

  it('prefers theme-agnostic icon when no theme matches', () => {
    const { container } = renderIcon({
      icons: [{ src: 'https://example.com/dark.svg', theme: 'dark' }, { src: 'https://example.com/universal.svg' }],
    });
    expect(getImg(container)).toHaveAttribute('src', 'https://example.com/universal.svg');
  });

  it('falls back to first icon when no theme-agnostic icon exists', () => {
    const { container } = renderIcon({
      icons: [
        { src: 'https://example.com/dark.svg', theme: 'dark' },
        { src: 'https://example.com/light.svg', theme: 'light' },
      ],
    });
    // Default theme is light mode in tests
    expect(getImg(container)).toHaveAttribute('src', 'https://example.com/light.svg');
  });
});
