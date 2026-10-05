import { useRef, useState } from 'react';
import {
  Button,
  CloseIcon,
  FormUI,
  Input,
  PlusIcon,
  PuzzleIcon,
  SimpleSelect,
  SimpleSelectOption,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { resolveIcon, sanitizeHref } from '../../common/utils/registryIcons';
import type { RegistryIcon } from '../types';

const PREVIEW_ICON_SIZE = 28;
const PREVIEW_BOX_SIZE = 44;

type IconTheme = 'Any' | 'Light' | 'Dark';

const themeToIcon = (theme: IconTheme) => (theme === 'Any' ? undefined : theme.toLowerCase());

const themeToOption = (theme?: string): IconTheme => {
  if (theme === 'light') return 'Light';
  if (theme === 'dark') return 'Dark';
  return 'Any';
};

const ThemeOptions = () => (
  <>
    <SimpleSelectOption value="Any">
      <FormattedMessage defaultMessage="Any" description="Theme-agnostic skill icon option" />
    </SimpleSelectOption>
    <SimpleSelectOption value="Light">
      <FormattedMessage defaultMessage="Light" description="Light mode skill icon option" />
    </SimpleSelectOption>
    <SimpleSelectOption value="Dark">
      <FormattedMessage defaultMessage="Dark" description="Dark mode skill icon option" />
    </SimpleSelectOption>
  </>
);

let lastRowId = 0;
const newRowId = () => `skill-icon-row-${(lastRowId += 1)}`;

const PreviewItem = ({
  isDark,
  icon,
  failed,
  onLoadError,
}: {
  isDark: boolean;
  icon: RegistryIcon | undefined;
  failed: boolean;
  onLoadError: (src: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const src = failed ? undefined : sanitizeHref(icon?.src);
  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: PREVIEW_BOX_SIZE,
          height: PREVIEW_BOX_SIZE,
          borderRadius: theme.borders.borderRadiusSm,
          backgroundColor: isDark ? '#1e1e1e' : '#ffffff',
          border: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
        }}
      >
        {src ? (
          <img
            src={src}
            alt=""
            referrerPolicy="no-referrer"
            onError={() => icon && onLoadError(icon.src)}
            css={{ width: PREVIEW_ICON_SIZE, height: PREVIEW_ICON_SIZE, objectFit: 'contain' }}
          />
        ) : (
          <PuzzleIcon
            aria-hidden
            css={{
              fontSize: PREVIEW_ICON_SIZE,
              color: isDark ? theme.colors.textPlaceholder : theme.colors.textSecondary,
            }}
          />
        )}
      </div>
      <Typography.Text color="secondary" size="sm">
        <FormattedMessage
          defaultMessage="{label} theme"
          description="Skill icon preview theme label"
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
  loadFailed,
  onChangeSrc,
  onChangeTheme,
  onRemove,
}: {
  icon: RegistryIcon;
  index: number;
  placeholder: string;
  selectWidth: number;
  loadFailed: boolean;
  onChangeSrc: (index: number, value: string) => void;
  onChangeTheme: (index: number, value: string) => void;
  onRemove: (index: number) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [localSrc, setLocalSrc] = useState(icon.src);
  const [syncedSrc, setSyncedSrc] = useState(icon.src);
  if (icon.src !== syncedSrc) {
    setSyncedSrc(icon.src);
    setLocalSrc(icon.src);
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column' }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <div css={{ flex: 1 }}>
          <Input
            componentId="mlflow.skill_registry.icon_editor.url"
            value={localSrc}
            onChange={(event) => setLocalSrc(event.target.value)}
            onBlur={() => {
              if (localSrc !== icon.src) onChangeSrc(index, localSrc);
            }}
            placeholder={placeholder}
            aria-label={intl.formatMessage(
              { defaultMessage: 'Icon URL {number}', description: 'Aria label for an existing skill icon URL' },
              { number: index + 1 },
            )}
            validationState={localSrc.trim() ? undefined : 'error'}
          />
        </div>
        <SimpleSelect
          id={`skill-icon-editor-theme-${index}`}
          componentId="mlflow.skill_registry.icon_editor.theme"
          aria-label={intl.formatMessage({
            defaultMessage: 'Theme',
            description: 'Aria label for a skill icon theme',
          })}
          value={themeToOption(icon.theme)}
          onChange={({ target }) => onChangeTheme(index, target.value)}
          css={{ width: selectWidth }}
        >
          <ThemeOptions />
        </SimpleSelect>
        <Tooltip
          componentId="mlflow.skill_registry.icon_editor.remove.tooltip"
          content={intl.formatMessage({
            defaultMessage: 'Remove icon',
            description: 'Tooltip for removing a skill icon',
          })}
        >
          <Button
            componentId="mlflow.skill_registry.icon_editor.remove"
            aria-label={intl.formatMessage({
              defaultMessage: 'Remove icon',
              description: 'Aria label for removing a skill icon',
            })}
            onClick={() => onRemove(index)}
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
            <FormattedMessage
              defaultMessage="Enter a valid URL"
              description="Error message when a skill icon URL is empty"
            />
          }
        />
      )}
      {localSrc.trim() && loadFailed && (
        <FormUI.Message
          type="error"
          message={
            <FormattedMessage
              defaultMessage="Image failed to load"
              description="Error message when a skill icon URL fails to load in the preview"
            />
          }
        />
      )}
    </div>
  );
};

export const SkillIconEditor = ({
  icons,
  onChange,
}: {
  icons: RegistryIcon[];
  onChange: (icons: RegistryIcon[]) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [draftUrl, setDraftUrl] = useState('');
  const [draftTheme, setDraftTheme] = useState<IconTheme>('Any');
  const [failedSrcs, setFailedSrcs] = useState<ReadonlySet<string>>(new Set());
  // Rows need an identity that survives URL edits: keyed by URL, a row remounted when its edit committed on blur,
  // swallowing a click on its Remove button.
  const rowIds = useRef<string[]>([]);
  while (rowIds.current.length < icons.length) rowIds.current.push(newRowId());
  rowIds.current.length = icons.length;
  const markFailed = (src: string) =>
    setFailedSrcs((current) => (current.has(src) ? current : new Set(current).add(src)));
  const selectWidth = theme.spacing.xl * 4;
  const placeholder = intl.formatMessage({
    defaultMessage: 'https://example.com/icon.svg',
    description: 'Placeholder for a skill icon URL',
  });

  const changeSrc = (index: number, value: string) => {
    onChange(icons.map((icon, iconIndex) => (iconIndex === index ? { ...icon, src: value } : icon)));
  };

  const changeTheme = (index: number, value: string) => {
    onChange(
      icons.map((icon, iconIndex) => {
        if (iconIndex !== index) return icon;
        const themeValue = themeToIcon(value as IconTheme);
        if (!themeValue) {
          const { theme: _theme, ...rest } = icon;
          return rest;
        }
        return { ...icon, theme: themeValue };
      }),
    );
  };

  const addIcon = () => {
    const src = draftUrl.trim();
    if (!src) return;
    const next: RegistryIcon = { src };
    const themeValue = themeToIcon(draftTheme);
    if (themeValue) next.theme = themeValue;
    onChange([...icons, next]);
    setDraftUrl('');
    setDraftTheme('Any');
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text bold>
        <FormattedMessage defaultMessage="Icons" description="Label for the skill icon editor" />
      </Typography.Text>
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
            <FormattedMessage defaultMessage="Icon URL" description="Column header for a skill icon URL" />
          </Typography.Text>
          <Typography.Text color="secondary" size="sm" css={{ width: selectWidth }}>
            <FormattedMessage defaultMessage="Theme" description="Column header for a skill icon theme" />
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
            onChangeSrc={changeSrc}
            onChangeTheme={changeTheme}
            onRemove={(iconIndex) => {
              rowIds.current.splice(iconIndex, 1);
              onChange(icons.filter((_, current) => current !== iconIndex));
            }}
          />
        ))}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <div css={{ flex: 1 }}>
            <Input
              componentId="mlflow.skill_registry.register_modal.icon_url"
              aria-label={intl.formatMessage({
                defaultMessage: 'Icon URL',
                description: 'Aria label for a skill icon URL',
              })}
              placeholder={placeholder}
              value={draftUrl}
              onChange={(event) => setDraftUrl(event.target.value)}
            />
          </div>
          <SimpleSelect
            id="skill-icon-editor-theme-draft"
            componentId="mlflow.skill_registry.register_modal.icon_theme"
            aria-label={intl.formatMessage({
              defaultMessage: 'Theme',
              description: 'Aria label for a skill icon theme',
            })}
            value={draftTheme}
            onChange={({ target }) => setDraftTheme(target.value as IconTheme)}
            css={{ width: selectWidth }}
          >
            <ThemeOptions />
          </SimpleSelect>
          <Tooltip
            componentId="mlflow.skill_registry.register_modal.icon_add.tooltip"
            content={intl.formatMessage({
              defaultMessage: 'Add icon',
              description: 'Tooltip for adding a skill icon',
            })}
          >
            <Button
              componentId="mlflow.skill_registry.register_modal.icon_add"
              aria-label={intl.formatMessage({
                defaultMessage: 'Add icon',
                description: 'Aria label for adding a skill icon',
              })}
              disabled={!draftUrl.trim()}
              onClick={addIcon}
            >
              <PlusIcon />
            </Button>
          </Tooltip>
        </div>
        <div>
          <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            <FormattedMessage defaultMessage="Preview" description="Label for the skill icon preview" />
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
                  onLoadError={markFailed}
                />
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
