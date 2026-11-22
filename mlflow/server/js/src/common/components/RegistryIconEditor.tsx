import { useRef, useState, type ReactNode } from 'react';
import {
  Button,
  CloseIcon,
  FormUI,
  Input,
  PlusIcon,
  SimpleSelect,
  SimpleSelectOption,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { useIconFallback } from '../hooks/useIconFallback';
import { resolveIcon, sanitizeHref, type RegistryIconImage } from '../utils/registryIcons';

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
      <FormattedMessage defaultMessage="Any" description="Registry icon editor option for a theme-agnostic icon" />
    </SimpleSelectOption>
    <SimpleSelectOption value="Light">
      <FormattedMessage defaultMessage="Light" description="Registry icon editor option for a light mode icon" />
    </SimpleSelectOption>
    <SimpleSelectOption value="Dark">
      <FormattedMessage defaultMessage="Dark" description="Registry icon editor option for a dark mode icon" />
    </SimpleSelectOption>
  </>
);

let lastId = 0;
const newId = () => `registry-icon-editor-${(lastId += 1)}`;

export interface RegistryIconFallback {
  icons: RegistryIconImage[];
  /** Tooltip for a preview that shows a fallback icon, e.g. "From server.json: <url>". */
  describe: (url: string) => string;
}

const PreviewItem = ({
  isDark,
  icon,
  failed,
  fallbackIcon,
  describeFallback,
  defaultIcon,
  componentId,
  onLoadError,
}: {
  isDark: boolean;
  icon: RegistryIconImage | undefined;
  /** The icon's URL already failed to load, in either preview. */
  failed: boolean;
  fallbackIcon: RegistryIconImage | undefined;
  describeFallback?: (url: string) => string;
  defaultIcon: ReactNode;
  componentId: string;
  onLoadError: (failedSrc: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const primarySrc = failed ? undefined : sanitizeHref(icon?.src);
  const fallbackSrc = sanitizeHref(fallbackIcon?.src);
  const { activeSrc, onError } = useIconFallback(primarySrc, fallbackSrc);

  const tooltipContent =
    activeSrc && activeSrc === primarySrc
      ? activeSrc
      : activeSrc && describeFallback
        ? describeFallback(activeSrc)
        : intl.formatMessage({
            defaultMessage: 'default',
            description: 'Registry icon editor tooltip for a preview showing the default icon',
          });

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      <Tooltip content={tooltipContent} componentId={`${componentId}.preview_src.tooltip`}>
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
            fontSize: PREVIEW_ICON_SIZE,
            color: isDark ? theme.colors.textPlaceholder : theme.colors.textSecondary,
          }}
        >
          {activeSrc ? (
            <img
              src={activeSrc}
              alt=""
              referrerPolicy="no-referrer"
              onError={() => {
                if (icon && activeSrc === primarySrc) onLoadError(icon.src);
                onError();
              }}
              css={{ width: PREVIEW_ICON_SIZE, height: PREVIEW_ICON_SIZE, objectFit: 'contain' }}
            />
          ) : (
            defaultIcon
          )}
        </div>
      </Tooltip>
      <Typography.Text color="secondary" size="sm">
        {isDark ? (
          <FormattedMessage defaultMessage="dark theme" description="Registry icon editor label for the dark preview" />
        ) : (
          <FormattedMessage
            defaultMessage="light theme"
            description="Registry icon editor label for the light preview"
          />
        )}
      </Typography.Text>
    </div>
  );
};

const IconRow = ({
  icon,
  index,
  placeholder,
  selectWidth,
  loadFailed,
  componentId,
  selectId,
  onChangeSrc,
  onChangeTheme,
  onRemove,
}: {
  icon: RegistryIconImage;
  index: number;
  placeholder: string;
  selectWidth: number;
  loadFailed: boolean;
  componentId: string;
  selectId: string;
  onChangeSrc: (index: number, value: string) => void;
  onChangeTheme: (index: number, value: ThemeOption) => void;
  onRemove: (index: number) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  // An edit stays a draft until blur commits it; otherwise the input shows the committed URL, so prop
  // changes need no syncing and the row never remounts under a pending click on Remove.
  const [draftSrc, setDraftSrc] = useState<string>();
  const src = draftSrc ?? icon.src;
  const removeLabel = intl.formatMessage({
    defaultMessage: 'Remove icon',
    description: 'Registry icon editor label for removing an icon',
  });

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <div css={{ flex: 1 }}>
          <Input
            componentId={`${componentId}.url`}
            value={src}
            onChange={(event) => setDraftSrc(event.target.value)}
            onBlur={() => {
              if (draftSrc === undefined) return;
              if (draftSrc !== icon.src) onChangeSrc(index, draftSrc);
              setDraftSrc(undefined);
            }}
            placeholder={placeholder}
            aria-label={intl.formatMessage(
              { defaultMessage: 'Icon URL {number}', description: 'Registry icon editor label for an icon URL' },
              { number: index + 1 },
            )}
            validationState={src.trim() ? undefined : 'error'}
          />
        </div>
        <SimpleSelect
          id={selectId}
          componentId={`${componentId}.theme`}
          aria-label={intl.formatMessage({
            defaultMessage: 'Theme',
            description: 'Registry icon editor label for an icon theme',
          })}
          value={themeValueToOption(icon.theme)}
          onChange={({ target }) => onChangeTheme(index, target.value as ThemeOption)}
          css={{ width: selectWidth }}
        >
          <ThemeSelectOptions />
        </SimpleSelect>
        <Tooltip content={removeLabel} componentId={`${componentId}.remove.tooltip`}>
          <Button
            componentId={`${componentId}.remove`}
            aria-label={removeLabel}
            onClick={() => onRemove(index)}
            dangerouslySetAntdProps={{ danger: true }}
          >
            <CloseIcon />
          </Button>
        </Tooltip>
      </div>
      {!src.trim() ? (
        <FormUI.Message
          type="error"
          message={
            <FormattedMessage
              defaultMessage="Enter a valid URL"
              description="Registry icon editor error for an empty icon URL"
            />
          }
        />
      ) : (
        loadFailed && (
          <FormUI.Message
            type="error"
            message={
              <FormattedMessage
                defaultMessage="Image failed to load"
                description="Registry icon editor error when an icon URL fails to load in the preview"
              />
            }
          />
        )
      )}
    </div>
  );
};

/**
 * Edits a registry entity's icons: one row per icon (URL + theme), a draft row to add one, and light and dark
 * previews. Shared by the MCP and Skill registries, which pass their own default icon and component IDs.
 */
export const RegistryIconEditor = <T extends RegistryIconImage>({
  icons,
  onChange,
  defaultIcon,
  componentId,
  fallback,
}: {
  icons: T[];
  onChange: (icons: T[]) => void;
  /** Shown in a preview when no icon (or fallback icon) applies to its theme. */
  defaultIcon: ReactNode;
  /** Prefix for the controls' component IDs, e.g. `mlflow.mcp_registry.icon_editor`. */
  componentId: string;
  /** Icons a preview falls back to when no explicit icon applies or loads, e.g. MCP's server.json icons. */
  fallback?: RegistryIconFallback;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [idPrefix] = useState(newId);
  const [draftUrl, setDraftUrl] = useState('');
  const [draftTheme, setDraftTheme] = useState<ThemeOption>('Any');
  const [failedSrcs, setFailedSrcs] = useState<ReadonlySet<string>>(new Set());
  // Rows need an identity that survives URL edits: keyed by URL, a row remounted when its edit committed on
  // blur, swallowing a click on its Remove button.
  const rowIds = useRef<string[]>([]);
  while (rowIds.current.length < icons.length) rowIds.current.push(newId());
  rowIds.current.length = icons.length;

  const placeholder = intl.formatMessage({
    defaultMessage: 'https://example.com/icon.svg',
    description: 'Registry icon editor placeholder for an icon URL',
  });
  const addLabel = intl.formatMessage({
    defaultMessage: 'Add icon',
    description: 'Registry icon editor label for adding an icon',
  });
  const selectWidth = theme.spacing.xl * 4;

  const withTheme = (icon: T, option: ThemeOption): T => {
    const { theme: _theme, ...rest } = icon;
    const value = themeOptionToValue(option);
    return (value ? { ...rest, theme: value } : rest) as T;
  };

  const changeSrc = (index: number, src: string) => {
    // Forget an earlier failure of the replaced URL, so returning to it later loads it again.
    const replaced = icons[index].src;
    setFailedSrcs((current) => {
      if (!current.has(replaced)) return current;
      const next = new Set(current);
      next.delete(replaced);
      return next;
    });
    onChange(icons.map((icon, current) => (current === index ? { ...icon, src } : icon)));
  };

  const changeTheme = (index: number, option: ThemeOption) => {
    onChange(icons.map((icon, current) => (current === index ? withTheme(icon, option) : icon)));
  };

  const remove = (index: number) => {
    rowIds.current.splice(index, 1);
    onChange(icons.filter((_, current) => current !== index));
  };

  const addDraft = () => {
    const src = draftUrl.trim();
    if (!src) return;
    onChange([...icons, withTheme({ src } as T, draftTheme)]);
    setDraftUrl('');
    setDraftTheme('Any');
  };

  const markFailed = (src: string) =>
    setFailedSrcs((current) => (current.has(src) ? current : new Set(current).add(src)));

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Typography.Text color="secondary" size="sm" css={{ flex: 1 }}>
          <FormattedMessage defaultMessage="Icon URL" description="Registry icon editor column header for icon URLs" />
        </Typography.Text>
        <Typography.Text color="secondary" size="sm" css={{ width: selectWidth }}>
          <FormattedMessage defaultMessage="Theme" description="Registry icon editor column header for icon themes" />
        </Typography.Text>
        <div css={{ width: theme.spacing.xl + theme.spacing.sm }} />
      </div>

      {icons.map((icon, index) => (
        <IconRow
          key={rowIds.current[index]}
          icon={icon}
          index={index}
          placeholder={placeholder}
          selectWidth={selectWidth}
          loadFailed={failedSrcs.has(icon.src)}
          componentId={componentId}
          selectId={`${idPrefix}-theme-${index}`}
          onChangeSrc={changeSrc}
          onChangeTheme={changeTheme}
          onRemove={remove}
        />
      ))}

      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <div css={{ flex: 1 }}>
          <Input
            componentId={`${componentId}.draft_url`}
            aria-label={intl.formatMessage({
              defaultMessage: 'New icon URL',
              description: 'Registry icon editor label for the URL of an icon being added',
            })}
            placeholder={placeholder}
            value={draftUrl}
            onChange={(event) => setDraftUrl(event.target.value)}
          />
        </div>
        <SimpleSelect
          id={`${idPrefix}-theme-draft`}
          componentId={`${componentId}.draft_theme`}
          aria-label={intl.formatMessage({
            defaultMessage: 'New icon theme',
            description: 'Registry icon editor label for the theme of an icon being added',
          })}
          value={draftTheme}
          onChange={({ target }) => setDraftTheme(target.value as ThemeOption)}
          css={{ width: selectWidth }}
        >
          <ThemeSelectOptions />
        </SimpleSelect>
        <Tooltip content={addLabel} componentId={`${componentId}.add.tooltip`}>
          <Button
            componentId={`${componentId}.add`}
            aria-label={addLabel}
            disabled={!draftUrl.trim()}
            onClick={addDraft}
          >
            <PlusIcon />
          </Button>
        </Tooltip>
      </div>

      <div>
        <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
          <FormattedMessage defaultMessage="Preview" description="Registry icon editor label for the icon previews" />
        </Typography.Text>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.md,
            padding: theme.spacing.md,
            backgroundColor: theme.colors.backgroundSecondary,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.general.borderRadiusBase,
          }}
        >
          {[false, true].map((isDark) => {
            const icon = resolveIcon(icons, isDark);
            return (
              <PreviewItem
                key={String(isDark)}
                isDark={isDark}
                icon={icon}
                failed={Boolean(icon && failedSrcs.has(icon.src))}
                fallbackIcon={resolveIcon(fallback?.icons, isDark)}
                describeFallback={fallback?.describe}
                defaultIcon={defaultIcon}
                componentId={componentId}
                onLoadError={markFailed}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
};
