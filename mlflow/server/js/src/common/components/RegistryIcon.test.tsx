import { describe, it, expect } from '@jest/globals';
import { fireEvent, render } from '@testing-library/react';
import { DesignSystemProvider, McpIcon } from '@databricks/design-system';
import { RegistryIcon, resolveIconSrc } from './RegistryIcon';
import type { RegistryIconImage } from '../utils/registryIcons';

const renderIcon = (props: Partial<React.ComponentProps<typeof RegistryIcon>>) =>
  render(
    <DesignSystemProvider>
      <RegistryIcon DefaultGlyph={McpIcon} {...props} />
    </DesignSystemProvider>,
  );

const getImg = (container: HTMLElement) => container.querySelector('img');
const getSvg = (container: HTMLElement) => container.querySelector('svg');

const light: RegistryIconImage = { src: 'https://example.com/light.svg', theme: 'light' };
const any: RegistryIconImage = { src: 'https://example.com/any.svg' };

describe('resolveIconSrc', () => {
  it('prefers primary icons and falls back when they have no match', () => {
    expect(resolveIconSrc([light], [any], false)).toBe(light.src);
    expect(resolveIconSrc([], [any], false)).toBe(any.src);
    expect(resolveIconSrc(null, null, false)).toBeUndefined();
  });
});

describe('RegistryIcon', () => {
  it('renders the default glyph when no icons are provided', () => {
    const { container } = renderIcon({});
    expect(getImg(container)).toBeNull();
    expect(getSvg(container)).toBeTruthy();
  });

  it('falls back to the default glyph when the image fails to load', () => {
    const { container } = renderIcon({ icons: [{ src: 'https://example.com/broken.svg' }] });
    fireEvent.error(getImg(container)!);
    expect(getImg(container)).toBeNull();
    expect(getSvg(container)).toBeTruthy();
  });

  it('falls back to fallbackIcons when the primary icon fails to load', () => {
    const { container } = renderIcon({
      icons: [{ src: 'https://example.com/broken.svg' }],
      fallbackIcons: [any],
    });
    fireEvent.error(getImg(container)!);
    expect(getImg(container)).toHaveAttribute('src', any.src);
  });

  it('ignores invalid javascript icon URLs', () => {
    const { container } = renderIcon({ icons: [{ src: `${'javascript'}:alert(1)` }] });
    expect(getImg(container)).toBeNull();
    expect(getSvg(container)).toBeTruthy();
  });
});
