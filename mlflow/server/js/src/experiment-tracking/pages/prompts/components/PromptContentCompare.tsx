import { Button, ExpandMoreIcon, Spacer, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { diffWords } from 'diff';
import { useMemo } from 'react';
import { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { getPromptContentTagValue } from '../utils';
import { PromptVersionMetadata } from './PromptVersionMetadata';
import { FormattedMessage } from 'react-intl';

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
  const baselineValue = useMemo(
    () => (baselineVersion ? getPromptContentTagValue(baselineVersion) : ''),
    [baselineVersion],
  );
  const comparedValue = useMemo(
    () => (comparedVersion ? getPromptContentTagValue(comparedVersion) : ''),
    [comparedVersion],
  );

  const diff = useMemo(() => diffWords(baselineValue ?? '', comparedValue ?? '') ?? [], [baselineValue, comparedValue]);

  const colors = {
    addedBackground: theme.colors.green300,
    removedBackground: theme.colors.red300,
  };

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
            description="TODO"
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
          <Tooltip componentId="TODO" content="Switch versions" side="top">
            <Button
              componentId="TODO"
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
