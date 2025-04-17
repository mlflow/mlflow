import { CircleIcon as DuboisCircleIcon, CheckCircleIcon, useDesignSystemTheme } from '@databricks/design-system';

export const getProtoField = (fieldName: string) => `${fieldName}`;

export function ReadyIcon() {
  const { theme } = useDesignSystemTheme();
  return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />;
}

type CircleIconProps = {
  type: 'FAILED' | 'PENDING' | 'READY';
};
