import { ESLintUtils } from '@typescript-eslint/utils';
import type { RuleWithMetaAndName } from '@typescript-eslint/utils/eslint-utils';

type BaseRuleConfig<MessageIds extends string> = Readonly<RuleWithMetaAndName<[], MessageIds>>;
type BaseRuleConfigMeta<MessageIds extends string> = Readonly<BaseRuleConfig<MessageIds>['meta']>;

type RuleConfigWithoutOptions<MessageIds extends string> = Omit<
  BaseRuleConfig<MessageIds>,
  'defaultOptions' | 'meta'
> & {
  meta: Omit<BaseRuleConfigMeta<MessageIds>, 'schema'>;
};

export const createRule = ESLintUtils.RuleCreator((name) => `@databricks/${name}`);

export function createRuleWithoutOptions<MessageIds extends string>(config: RuleConfigWithoutOptions<MessageIds>) {
  return createRule({
    ...config,
    meta: { ...config.meta, schema: [] },
    defaultOptions: [],
  });
}
