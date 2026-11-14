import type { ElementType } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import type { RegistryIconImage } from '../utils/registryIcons';
import { resolveIcon, sanitizeHref } from '../utils/registryIcons';
import { useIconFallback } from '../hooks/useIconFallback';

const ICON_SIZE = 16;

export const resolveIconSrc = <T extends RegistryIconImage>(
  icons?: T[] | null,
  fallbackIcons?: T[] | null,
  isDarkMode?: boolean,
): string | undefined => resolveIcon(icons, isDarkMode)?.src ?? resolveIcon(fallbackIcons, isDarkMode)?.src;

export const RegistryIcon = <T extends RegistryIconImage>({
  icons,
  fallbackIcons,
  name,
  css: cssProp,
  DefaultGlyph,
}: {
  icons?: T[] | null;
  fallbackIcons?: T[] | null;
  name?: string;
  css?: Record<string, unknown>;
  DefaultGlyph: ElementType;
}) => {
  const { theme } = useDesignSystemTheme();
  const primarySrc = sanitizeHref(resolveIcon(icons, theme.isDarkMode)?.src);
  const fallbackSrc = sanitizeHref(resolveIcon(fallbackIcons, theme.isDarkMode)?.src);
  const { activeSrc, onError } = useIconFallback(primarySrc, fallbackSrc);
  const glyphStyles = {
    flexShrink: 0,
    color: theme.colors.textSecondary,
    width: ICON_SIZE,
    height: ICON_SIZE,
    ...cssProp,
  };

  if (activeSrc) {
    return (
      <img
        src={activeSrc}
        alt={name || ''}
        referrerPolicy="no-referrer"
        onError={onError}
        css={{
          width: ICON_SIZE,
          height: ICON_SIZE,
          objectFit: 'contain',
          flexShrink: 0,
          color: theme.colors.textSecondary,
          ...cssProp,
        }}
      />
    );
  }

  return <DefaultGlyph aria-hidden css={glyphStyles} />;
};
