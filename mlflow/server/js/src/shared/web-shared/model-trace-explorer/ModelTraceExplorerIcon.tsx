import {
  ModelsIcon,
  ConnectIcon,
  FileDocumentIcon,
  useDesignSystemTheme,
  SortUnsortedIcon,
  QuestionMarkIcon,
  CodeIcon,
  FunctionIcon,
  NumbersIcon,
  SearchIcon,
  WrenchIcon,
  UserSparkleIcon,
  ChainIcon,
  UserIcon,
  GearIcon,
  SaveIcon,
} from '@databricks/design-system';

import { ModelIconType } from './ModelTrace.types';

export const ModelTraceExplorerIcon = ({
  type = ModelIconType.CONNECT,
  // tooltips have inverted colors so the icon should match it
  isInTooltip = false,
  hasException = false,
  isRootSpan = false,
}: {
  type?: ModelIconType;
  isInTooltip?: boolean;
  hasException?: boolean;
  isRootSpan?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  // base icon colors depending on span attributes
  let iconColor: 'ai' | 'danger' | undefined;
  if (isRootSpan) {
    iconColor = 'ai';
  } else if (hasException) {
    iconColor = 'danger';
  }

  const iconMap = {
    [ModelIconType.MODELS]: <ModelsIcon color={iconColor} />,
    [ModelIconType.DOCUMENT]: <FileDocumentIcon color={iconColor} />,
    [ModelIconType.CONNECT]: <ConnectIcon color={iconColor} />,
    [ModelIconType.CODE]: <CodeIcon color={iconColor} />,
    [ModelIconType.FUNCTION]: <FunctionIcon color={iconColor} />,
    [ModelIconType.NUMBERS]: <NumbersIcon color={iconColor} />,
    [ModelIconType.SEARCH]: <SearchIcon color={iconColor} />,
    [ModelIconType.SORT]: <SortUnsortedIcon color={iconColor} />,
    [ModelIconType.UNKNOWN]: <QuestionMarkIcon color={iconColor} />,
    [ModelIconType.WRENCH]: <WrenchIcon color={iconColor} />,
    [ModelIconType.AGENT]: <UserSparkleIcon color={iconColor} />,
    [ModelIconType.CHAIN]: <ChainIcon color={iconColor} />,
    [ModelIconType.USER]: <UserIcon color={iconColor} />,
    [ModelIconType.SYSTEM]: <GearIcon color={iconColor} />,
    [ModelIconType.SAVE]: <SaveIcon color={iconColor} />,
  };

  // custom colors depending on span type
  // these are not official props on the
  // icon components, so they must be set
  // via the `css` prop on the parent
  let color: string = theme.colors.actionDefaultIconDefault;
  let tooltipColor: string = theme.colors.actionPrimaryIcon;
  let backgroundColor: string = theme.colors.backgroundSecondary;
  switch (type) {
    case ModelIconType.SEARCH:
      color = theme.colors.textValidationSuccess;
      tooltipColor = theme.colors.green500;
      backgroundColor = theme.isDarkMode ? theme.colors.green800 : theme.colors.green100;
      break;
    case ModelIconType.MODELS:
      color = theme.isDarkMode ? theme.colors.blue500 : theme.colors.turquoise;
      tooltipColor = theme.isDarkMode ? theme.colors.turquoise : theme.colors.blue500;
      backgroundColor = theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100;
      break;
    case ModelIconType.WRENCH:
      color = theme.isDarkMode ? theme.colors.red500 : theme.colors.red700;
      tooltipColor = theme.isDarkMode ? theme.colors.red700 : theme.colors.red500;
      backgroundColor = theme.isDarkMode ? theme.colors.red800 : theme.colors.red100;
      break;
  }

  return (
    <div
      css={{
        position: 'relative',
        width: theme.general.iconSize,
        height: theme.general.iconSize,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: theme.borders.borderRadiusSm,
        background: isInTooltip ? theme.colors.tooltipBackgroundTooltip : backgroundColor,
        color: isInTooltip ? tooltipColor : color,
        svg: { width: theme.general.iconFontSize, height: theme.general.iconFontSize },
        flexShrink: 0,
      }}
    >
      {hasException && (
        <div
          css={{
            position: 'absolute',
            top: -theme.spacing.xs,
            right: -theme.spacing.xs,
            height: theme.spacing.sm,
            width: theme.spacing.sm,
            borderRadius: theme.borders.borderRadiusSm,
            backgroundColor: theme.colors.actionDangerPrimaryBackgroundDefault,
            zIndex: 5,
          }}
        />
      )}
      {iconMap[type]}
    </div>
  );
};
