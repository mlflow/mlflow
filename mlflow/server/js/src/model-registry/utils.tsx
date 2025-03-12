import {
  CircleIcon as DuboisCircleIcon,
  CheckCircleIcon,
  useDesignSystemTheme,
  WarningFillIcon,
} from '@databricks/design-system';

export const getProtoField = (fieldName: string) => `${fieldName}`;

export function ReadyIcon() {
  const { theme } = useDesignSystemTheme();
  return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />;
}

export function FailedIcon() {
  const { theme } = useDesignSystemTheme();
  return <WarningFillIcon css={{ color: theme.colors.textValidationDanger }} />;
}

type CircleIconProps = {
  type: 'FAILED' | 'PENDING' | 'READY';
};

export function CircleIcon({ type }: CircleIconProps) {
  const { theme } = useDesignSystemTheme();
  let color;
  switch (type) {
    case 'FAILED': {
      color = theme.colors.textValidationDanger;
      break;
    }
    case 'PENDING': {
      color = theme.colors.yellow400; // textValidationWarning was too dark/red
      break;
    }
    case 'READY':
    default: {
      color = theme.colors.green500;
      break;
    }
  }
  return <DuboisCircleIcon css={{ color, fontSize: 16 }} />;
}
