import { describe, it, expect } from '@jest/globals';
import { render, fireEvent } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { MCPServerIcon, resolveIconSrc } from './MCPServerIcon';
import type { MCPIcon } from '../types';

const renderIcon = (props: React.ComponentProps<typeof MCPServerIcon>) =>
  render(
    <DesignSystemProvider>
      <MCPServerIcon {...props} />
    </DesignSystemProvider>,
  );

const getImg = (container: HTMLElement) => container.querySelector('img');
const getSvg = (container: HTMLElement) => container.querySelector('svg');

// icons = explicitly added via icon editor (server-level)
// fallbackIcons = from server.json

const sjLight: MCPIcon = { src: 'https://example.com/sj-light.svg', theme: 'light' };
const sjDark: MCPIcon = { src: 'https://example.com/sj-dark.svg', theme: 'dark' };
const sjAny: MCPIcon = { src: 'https://example.com/sj-any.svg' };

const addedLight: MCPIcon = { src: 'https://example.com/added-light.svg', theme: 'light' };
const addedDark: MCPIcon = { src: 'https://example.com/added-dark.svg', theme: 'dark' };
const addedAny: MCPIcon = { src: 'https://example.com/added-any.svg' };

describe('resolveIconSrc', () => {
  describe('server.json icons only (no icons added via editor)', () => {
    it.each<{ name: string; sjIcons: MCPIcon[]; isDarkMode: boolean; expected: string | undefined }>([
      { name: 'light only, light mode', sjIcons: [sjLight], isDarkMode: false, expected: sjLight.src },
      { name: 'light only, dark mode', sjIcons: [sjLight], isDarkMode: true, expected: undefined },
      { name: 'dark only, dark mode', sjIcons: [sjDark], isDarkMode: true, expected: sjDark.src },
      { name: 'dark only, light mode', sjIcons: [sjDark], isDarkMode: false, expected: undefined },
      { name: 'both, light mode', sjIcons: [sjLight, sjDark], isDarkMode: false, expected: sjLight.src },
      { name: 'both, dark mode', sjIcons: [sjLight, sjDark], isDarkMode: true, expected: sjDark.src },
      { name: 'any (no theme), light mode', sjIcons: [sjAny], isDarkMode: false, expected: sjAny.src },
      { name: 'any (no theme), dark mode', sjIcons: [sjAny], isDarkMode: true, expected: sjAny.src },
      { name: 'none', sjIcons: [], isDarkMode: false, expected: undefined },
    ])('$name', ({ sjIcons, isDarkMode, expected }) => {
      expect(resolveIconSrc(undefined, sjIcons, isDarkMode)).toBe(expected);
    });
  });

  describe('fallback: added icons → server.json icons', () => {
    it.each<{ name: string; added: MCPIcon[]; sjIcons: MCPIcon[]; isDarkMode: boolean; expected: string | undefined }>([
      {
        name: 'added light only, sj has dark, dark mode → sj dark',
        added: [addedLight],
        sjIcons: [sjDark],
        isDarkMode: true,
        expected: sjDark.src,
      },
      {
        name: 'added light only, sj has dark, light mode → added light',
        added: [addedLight],
        sjIcons: [sjDark],
        isDarkMode: false,
        expected: addedLight.src,
      },
      {
        name: 'added dark only, sj has any, light mode → sj any',
        added: [addedDark],
        sjIcons: [sjAny],
        isDarkMode: false,
        expected: sjAny.src,
      },
      {
        name: 'added match exists, sj ignored',
        added: [addedLight, addedDark],
        sjIcons: [sjAny],
        isDarkMode: true,
        expected: addedDark.src,
      },
      { name: 'nothing added, sj any → sj any', added: [], sjIcons: [sjAny], isDarkMode: false, expected: sjAny.src },
      {
        name: 'nothing added, nothing in sj → undefined',
        added: [],
        sjIcons: [],
        isDarkMode: false,
        expected: undefined,
      },
    ])('$name', ({ added, sjIcons, isDarkMode, expected }) => {
      expect(resolveIconSrc(added, sjIcons, isDarkMode)).toBe(expected);
    });
  });
});

describe('MCPServerIcon', () => {
  it('renders default icon when no icons provided', () => {
    const { container } = renderIcon({});
    expect(getImg(container)).toBeNull();
    expect(getSvg(container)).toBeTruthy();
  });

  it('falls back to default icon when img fails to load', () => {
    const { container } = renderIcon({ icons: [{ src: 'https://example.com/broken.svg' }] });
    fireEvent.error(getImg(container)!);
    expect(getImg(container)).toBeNull();
    expect(getSvg(container)).toBeTruthy();
  });

  it('falls back to fallbackIcons when primary icon fails to load', () => {
    const { container } = renderIcon({
      icons: [{ src: 'https://example.com/broken.svg' }],
      fallbackIcons: [sjAny],
    });
    fireEvent.error(getImg(container)!);
    expect(getImg(container)).toHaveAttribute('src', sjAny.src);
  });

  it('falls back to default when both primary and fallback fail to load', () => {
    const { container } = renderIcon({
      icons: [{ src: 'https://example.com/broken.svg' }],
      fallbackIcons: [{ src: 'https://example.com/also-broken.svg' }],
    });
    fireEvent.error(getImg(container)!);
    expect(getImg(container)).toHaveAttribute('src', 'https://example.com/also-broken.svg');
    fireEvent.error(getImg(container)!);
    expect(getImg(container)).toBeNull();
    expect(getSvg(container)).toBeTruthy();
  });

  it('uses fallbackIcons when primary icons have no match', () => {
    const { container } = renderIcon({ icons: [], fallbackIcons: [sjAny] });
    expect(getImg(container)).toHaveAttribute('src', sjAny.src);
  });
});
