import { useEffect, useState } from 'react';
import { McpIcon, useDesignSystemTheme } from '@databricks/design-system';
import type { MCPIcon as MCPIconType } from '../types';
import { mcpIconStyles } from '../styles';

const ICON_SIZE = 16;

const resolveIconSrc = (icons?: MCPIconType[], isDarkMode?: boolean): string | undefined => {
  if (!icons?.length) return undefined;
  const preferred = isDarkMode ? 'dark' : 'light';
  return icons.find((i) => i.theme === preferred)?.src ?? icons.find((i) => !i.theme)?.src ?? icons[0]?.src;
};

export const MCPServerIcon = ({
  icons,
  name,
  css: cssProp,
}: {
  icons?: MCPIconType[];
  name?: string;
  css?: Record<string, unknown>;
}) => {
  const { theme } = useDesignSystemTheme();
  const [imgFailed, setImgFailed] = useState(false);
  const iconSrc = resolveIconSrc(icons, theme.isDarkMode);

  useEffect(() => {
    setImgFailed(false);
  }, [iconSrc]);

  if (iconSrc && !imgFailed) {
    return (
      <img
        src={iconSrc}
        alt={name || ''}
        referrerPolicy="no-referrer"
        onError={() => setImgFailed(true)}
        css={{
          width: ICON_SIZE,
          height: ICON_SIZE,
          objectFit: 'contain',
          ...mcpIconStyles(theme),
          ...cssProp,
        }}
      />
    );
  }

  return <McpIcon aria-hidden css={{ width: ICON_SIZE, height: ICON_SIZE, ...mcpIconStyles(theme), ...cssProp }} />;
};
