import { useEffect, useState } from 'react';
import {
  Button,
  CloseIcon,
  FormUI,
  Input,
  McpIcon,
  PlusIcon,
  SimpleSelect,
  SimpleSelectOption,
  Space,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { MCPIcon } from '../types';
import { previewRowStyles } from '../styles';
import { resolveIcon, sanitizeHref } from '../utils';
import { useIconFallback } from '../hooks/useIconFallback';

const PREVIEW_ICON_SIZE = 28;
const PREVIEW_BOX_SIZE = 44;

type ThemeOption = 'Any' | 'Light' | 'Dark';

const themeOptionToValue = (option: ThemeOption): string | undefined =>
  option === 'Any' ? undefined : option.toLowerCase();

const themeValueToOption = (theme?: string): ThemeOption => {
  if (theme === 'light') return 'Light';
  if (theme === 'dark') return 'Dark';
  return 'Any';
};

const ThemeSelectOptions = () => (
  <>
    <SimpleSelectOption value="Any">
      <FormattedMessage defaultMessage="Any" description="Theme-agnostic icon option" />
    </SimpleSelectOption>
    <SimpleSelectOption value="Light">
      <FormattedMessage defaultMessage="Light" description="Light mode icon option" />
    </SimpleSelectOption>
    <SimpleSelectOption value="Dark">
      <FormattedMessage defaultMessage="Dark" description="Dark mode icon option" />
    </SimpleSelectOption>
  </>
);

type IconSource = 'explicit' | 'server-json' | 'default';

const PreviewItem = ({
  isDark,
  icon,
  fallbackIcon,
  onLoadError,
}: {
  isDark: boolean;
  icon: MCPIcon | undefined;
  fallbackIcon: MCPIcon | undefined;
  onLoadError?: (failedSrc: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const sanitizedPrimary = sanitizeHref(icon?.src);
  const sanitizedFallback = sanitizeHref(fallbackIcon?.src);
  const { activeSrc, onError: onIconError } = useIconFallback(sanitizedPrimary, sanitizedFallback);

  const source: IconSource =
    activeSrc === sanitizedPrimary ? 'explicit' : activeSrc === sanitizedFallback ? 'server-json' : 'default';

  const tooltipContent =
    source === 'explicit'
      ? activeSrc
      : source === 'server-json'
        ? intl.formatMessage(
            { defaultMessage: 'From server.json: {url}', description: 'Tooltip for server.json icon' },
            { url: activeSrc },
          )
        : intl.formatMessage({ defaultMessage: 'default', description: 'Tooltip for default fallback icon' });

  const iconBox = (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: PREVIEW_BOX_SIZE,
        height: PREVIEW_BOX_SIZE,
        borderRadius: theme.borders.borderRadiusSm,
        // Intentionally hard-coded: simulates target light/dark backgrounds
        backgroundColor: isDark ? '#1e1e1e' : '#ffffff',
        border: `1px solid ${theme.colors.border}`,
        flexShrink: 0,
      }}
    >
      {activeSrc ? (
        <img
          src={activeSrc}
          alt=""
          referrerPolicy="no-referrer"
          onError={() => {
            if (activeSrc === sanitizedPrimary) {
              onLoadError?.(icon?.src ?? '');
            }
            onIconError();
          }}
          css={{ width: PREVIEW_ICON_SIZE, height: PREVIEW_ICON_SIZE, objectFit: 'contain' }}
        />
      ) : (
        <McpIcon
          aria-hidden
          css={{
            fontSize: PREVIEW_ICON_SIZE,
            width: PREVIEW_ICON_SIZE,
            height: PREVIEW_ICON_SIZE,
            color: isDark ? theme.colors.textPlaceholder : theme.colors.textSecondary,
          }}
        />
      )}
    </div>
  );

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      <Tooltip content={tooltipContent} componentId="mlflow.mcp_registry.icon_editor.preview_src.tooltip">
        {iconBox}
      </Tooltip>
      <Typography.Text color="secondary" size="sm">
        <FormattedMessage
          defaultMessage="{label} theme"
          description="Preview item theme label"
          values={{ label: isDark ? 'dark' : 'light' }}
        />
      </Typography.Text>
    </div>
  );
};

const IconRow = ({
  icon,
  index,
  placeholder,
  selectWidth,
  hasError,
  onChangeSrc,
  onChangeTheme,
  onRemove,
}: {
  icon: MCPIcon;
  index: number;
  placeholder: string;
  selectWidth: number;
  hasError: boolean;
  onChangeSrc: (index: number, value: string) => void;
  onChangeTheme: (index: number, value: string) => void;
  onRemove: (index: number) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [localSrc, setLocalSrc] = useState(icon.src);

  useEffect(() => {
    setLocalSrc(icon.src);
  }, [icon.src]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <div css={{ flex: 1 }}>
          <Input
            componentId="mlflow.mcp_registry.icon_editor.url"
            value={localSrc}
            onChange={(e) => setLocalSrc(e.target.value)}
            onBlur={() => {
              if (localSrc !== icon.src) {
                onChangeSrc(index, localSrc);
              }
            }}
            placeholder={placeholder}
            validationState={hasError || !localSrc.trim() ? 'error' : undefined}
          />
        </div>
        <SimpleSelect
          id={`mcp-registry-icon-editor-theme-${index}`}
          componentId="mlflow.mcp_registry.icon_editor.theme"
          value={themeValueToOption(icon.theme)}
          onChange={({ target }) => onChangeTheme(index, target.value)}
          css={{ width: selectWidth }}
        >
          <ThemeSelectOptions />
        </SimpleSelect>
        <Tooltip
          content={intl.formatMessage({
            defaultMessage: 'Remove icon',
            description: 'Tooltip for remove icon button in MCP server icon editor',
          })}
          componentId="mlflow.mcp_registry.icon_editor.remove.tooltip"
        >
          <Button
            componentId="mlflow.mcp_registry.icon_editor.remove"
            onClick={() => onRemove(index)}
            aria-label={intl.formatMessage({
              defaultMessage: 'Remove icon',
              description: 'Aria label for remove icon button in MCP server icon editor',
            })}
            dangerouslySetAntdProps={{ danger: true }}
          >
            <CloseIcon />
          </Button>
        </Tooltip>
      </div>
      {!localSrc.trim() && (
        <FormUI.Message
          type="error"
          message={
            <FormattedMessage defaultMessage="Enter a valid URL" description="Error message when icon URL is empty" />
          }
        />
      )}
      {localSrc.trim() && hasError && (
        <FormUI.Message
          type="error"
          message={
            <FormattedMessage
              defaultMessage="Image failed to load"
              description="Error message when icon URL fails to load in preview"
            />
          }
        />
      )}
    </div>
  );
};

export const IconEditor = ({
  icons,
  onChange,
  serverJsonIcons,
}: {
  icons: MCPIcon[];
  onChange: (icons: MCPIcon[]) => void;
  serverJsonIcons?: MCPIcon[];
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [draftUrl, setDraftUrl] = useState('');
  const [draftTheme, setDraftTheme] = useState<ThemeOption>('Any');
  const [failedSrcs, setFailedSrcs] = useState<Set<string>>(new Set());

  const sjIcons = serverJsonIcons ?? [];

  const handleSrcChange = (index: number, value: string) => {
    setFailedSrcs((prev) => {
      if (!prev.has(icons[index].src)) return prev;
      const next = new Set(prev);
      next.delete(icons[index].src);
      return next;
    });
    onChange(icons.map((icon, i) => (i === index ? { ...icon, src: value } : icon)));
  };

  const handleThemeChange = (index: number, value: string) => {
    const updated = icons.map((icon, i) => {
      if (i !== index) return icon;
      const themeVal = themeOptionToValue(value as ThemeOption);
      if (themeVal) return { ...icon, theme: themeVal };
      const { theme: _, ...rest } = icon;
      return rest;
    });
    onChange(updated);
  };

  const handleAddDraft = () => {
    const trimmed = draftUrl.trim();
    if (!trimmed) return;
    const newIcon: MCPIcon = { src: trimmed };
    const themeVal = themeOptionToValue(draftTheme);
    if (themeVal) {
      newIcon.theme = themeVal;
    }
    onChange([...icons, newIcon]);
    setDraftUrl('');
    setDraftTheme('Any');
  };

  const handleRemove = (index: number) => {
    onChange(icons.filter((_, i) => i !== index));
  };

  const handleLoadError = (failedSrc: string) => {
    setFailedSrcs((prev) => {
      if (prev.has(failedSrc)) return prev;
      const next = new Set(prev);
      next.add(failedSrc);
      return next;
    });
  };

  const placeholder = intl.formatMessage({
    defaultMessage: 'https://example.com/icon.svg',
    description: 'Placeholder for icon URL input in MCP server icon editor',
  });

  const selectWidth = theme.spacing.xl * 4;
  const buttonSpacerWidth = theme.spacing.xl + theme.spacing.sm;

  return (
    <Space
      direction="vertical"
      size="small"
      css={{
        width: '100%',
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Text color="secondary" size="sm" css={{ flex: 1 }}>
          <FormattedMessage defaultMessage="Icon URL" description="Column header for icon URL" />
        </Typography.Text>
        <Typography.Text color="secondary" size="sm" css={{ width: selectWidth }}>
          <FormattedMessage defaultMessage="Theme" description="Column header for icon theme" />
        </Typography.Text>
        <div css={{ width: buttonSpacerWidth }} />
      </div>

      {/* Confirmed rows */}
      {icons.map((icon, index) => (
        <IconRow
          key={`${icon.src}-${icon.theme ?? 'Any'}-${index}`}
          icon={icon}
          index={index}
          placeholder={placeholder}
          selectWidth={selectWidth}
          hasError={failedSrcs.has(icon.src)}
          onChangeSrc={handleSrcChange}
          onChangeTheme={handleThemeChange}
          onRemove={handleRemove}
        />
      ))}

      {/* Draft row */}
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <div css={{ flex: 1 }}>
          <Input
            componentId="mlflow.mcp_registry.icon_editor.draft_url"
            value={draftUrl}
            onChange={(e) => setDraftUrl(e.target.value)}
            placeholder={placeholder}
          />
        </div>
        <SimpleSelect
          id="mcp-registry-icon-editor-theme-draft"
          componentId="mlflow.mcp_registry.icon_editor.draft_theme"
          value={draftTheme}
          onChange={({ target }) => setDraftTheme(target.value as ThemeOption)}
          css={{ width: selectWidth }}
        >
          <ThemeSelectOptions />
        </SimpleSelect>
        <Tooltip
          content={intl.formatMessage({
            defaultMessage: 'Add icon',
            description: 'Tooltip for add icon button in MCP server icon editor',
          })}
          componentId="mlflow.mcp_registry.icon_editor.add.tooltip"
        >
          <Button
            componentId="mlflow.mcp_registry.icon_editor.add"
            onClick={handleAddDraft}
            disabled={!draftUrl.trim()}
            aria-label={intl.formatMessage({
              defaultMessage: 'Add icon',
              description: 'Aria label for add icon button in MCP server icon editor',
            })}
          >
            <PlusIcon />
          </Button>
        </Tooltip>
      </div>

      {/* Preview */}
      <div>
        <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
          <FormattedMessage defaultMessage="Preview" description="Preview label in MCP server icon editor" />
        </Typography.Text>
        <div css={previewRowStyles(theme)}>
          <PreviewItem
            isDark={false}
            icon={resolveIcon(icons, false)}
            fallbackIcon={resolveIcon(sjIcons, false)}
            onLoadError={handleLoadError}
          />
          <PreviewItem
            isDark
            icon={resolveIcon(icons, true)}
            fallbackIcon={resolveIcon(sjIcons, true)}
            onLoadError={handleLoadError}
          />
        </div>
      </div>
    </Space>
  );
};
