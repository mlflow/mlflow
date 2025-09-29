import { Button, ExpandMoreIcon, Spacer, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import type { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { getPromptContentTagValue } from '../utils';
import { PromptVersionMetadata } from './PromptVersionMetadata';
import { FormattedMessage, useIntl } from 'react-intl';
import { diffWords } from '../diff';

export const PromptContentCompare = ({
  baselineVersion,
  comparedVersion,
  onSwitchSides,
  onEditVersion,
  registeredPrompt,
  aliasesByVersion,
  showEditAliasesModal,
}: {
  baselineVersion?: RegisteredPromptVersion;
  comparedVersion?: RegisteredPromptVersion;
  onSwitchSides: () => void;
  onEditVersion: (version?: RegisteredPromptVersion) => void;
  registeredPrompt?: RegisteredPrompt;
  aliasesByVersion: Record<string, string[]>;
  showEditAliasesModal?: (versionNumber: string) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const baselineValue = useMemo(
    () => (baselineVersion ? getPromptContentTagValue(baselineVersion) : ''),
    [baselineVersion],
  );
  const comparedValue = useMemo(
    () => (comparedVersion ? getPromptContentTagValue(comparedVersion) : ''),
    [comparedVersion],
  );

  const diff = useMemo(() => diffWords(baselineValue ?? '', comparedValue ?? '') ?? [], [baselineValue, comparedValue]);

  const colors = useMemo(
    () => ({
      addedBackground: theme.isDarkMode ? theme.colors.green700 : theme.colors.green300,
      removedBackground: theme.isDarkMode ? theme.colors.red700 : theme.colors.red300,
    }),
    [theme],
  );

  return (
    <div
      css={{
        flex: 1,
        padding: theme.spacing.md,
        paddingTop: 0,
        borderRadius: theme.borders.borderRadiusSm,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography.Title level={3}>
          <FormattedMessage
            defaultMessage="Comparing version {baseline} with version {compared}"
            description="Label for comparing prompt versions in the prompt comparison view. Variables {baseline} and {compared} are numeric version numbers being compared."
            values={{
              baseline: baselineVersion?.version,
              compared: comparedVersion?.version,
            }}
          />
        </Typography.Title>
      </div>
      <Spacer shrinks={false} />
      <div css={{ display: 'flex' }}>
        <div css={{ flex: 1 }}>
          <PromptVersionMetadata
            aliasesByVersion={aliasesByVersion}
            onEditVersion={onEditVersion}
            registeredPrompt={registeredPrompt}
            registeredPromptVersion={baselineVersion}
            showEditAliasesModal={showEditAliasesModal}
            isBaseline
          />
        </div>
        <div css={{ paddingLeft: theme.spacing.sm, paddingRight: theme.spacing.sm }}>
          <div css={{ width: theme.general.heightSm }} />
        </div>
        <div css={{ flex: 1 }}>
          <PromptVersionMetadata
            aliasesByVersion={aliasesByVersion}
            onEditVersion={onEditVersion}
            registeredPrompt={registeredPrompt}
            registeredPromptVersion={comparedVersion}
            showEditAliasesModal={showEditAliasesModal}
          />
        </div>
      </div>
      <Spacer shrinks={false} />
      <div css={{ display: 'flex', flex: 1, overflow: 'auto', alignItems: 'flex-start' }}>
        <div
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            padding: theme.spacing.md,
            flex: 1,
          }}
        >
          <Typography.Text
            css={{
              whiteSpace: 'pre-wrap',
            }}
          >
            {baselineValue || 'Empty'}
          </Typography.Text>
        </div>
        <div css={{ paddingLeft: theme.spacing.sm, paddingRight: theme.spacing.sm }}>
          <Tooltip
            componentId="mlflow.prompts.details.switch_sides.tooltip"
            content={
              <FormattedMessage
                defaultMessage="Switch sides"
                description="A label for button used to switch prompt versions when in side-by-side comparison view"
              />
            }
            side="top"
          >
            <Button
              aria-label={intl.formatMessage({
                defaultMessage: 'Switch sides',
                description: 'A label for button used to switch prompt versions when in side-by-side comparison view',
              })}
              componentId="mlflow.prompts.details.switch_sides"
              icon={<ExpandMoreIcon css={{ svg: { rotate: '90deg' } }} />}
              onClick={onSwitchSides}
            />
          </Tooltip>
        </div>

        <div
          css={{
            backgroundColor: theme.colors.backgroundSecondary,
            padding: theme.spacing.md,
            flex: 1,
          }}
        >
          <Typography.Text
            css={{
              whiteSpace: 'pre-wrap',
            }}
          >
            {diff.map((part, index) => (
              <span
                key={index}
                css={{
                  backgroundColor: part.added
                    ? colors.addedBackground
                    : part.removed
                    ? colors.removedBackground
                    : undefined,
                  textDecoration: part.removed ? 'line-through' : 'none',
                }}
              >
                {part.value}
              </span>
            ))}
          </Typography.Text>
        </div>
      </div>
    </div>
  );
};
