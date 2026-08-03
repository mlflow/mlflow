import type { ComponentType } from 'react';

import { z } from 'zod';
import { createComponentImplementation, type ReactComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  CalendarEventIcon,
  CalendarIcon,
  CameraIcon,
  CheckIcon,
  CircleOutlineIcon,
  CloseIcon,
  DangerIcon,
  DownloadIcon,
  FolderIcon,
  GearIcon,
  HomeIcon,
  ImageIcon,
  InfoIcon,
  ListIcon,
  LockIcon,
  LockUnlockedIcon,
  MailIcon,
  MenuIcon,
  NotificationIcon,
  NotificationOffIcon,
  OverflowIcon,
  PauseIcon,
  PencilIcon,
  PinIcon,
  PlayIcon,
  PlusIcon,
  QuestionMarkIcon,
  RefreshIcon,
  SearchIcon,
  SendIcon,
  ShareIcon,
  StarIcon,
  StopIcon,
  TagIcon,
  TrashIcon,
  UploadIcon,
  UserCircleIcon,
  UserIcon,
  VisibleIcon,
  VisibleOffIcon,
  WarningIcon,
} from '@databricks/design-system';

// Maps icon names to Databricks Design System icons so glyphs stay visually
// native to the rest of MLflow.
const DS_ICON_BY_NAME = {
  accountCircle: UserCircleIcon,
  add: PlusIcon,
  arrowBack: ArrowLeftIcon,
  arrowForward: ArrowRightIcon,
  calendarToday: CalendarIcon,
  camera: CameraIcon,
  check: CheckIcon,
  close: CloseIcon,
  delete: TrashIcon,
  download: DownloadIcon,
  edit: PencilIcon,
  event: CalendarEventIcon,
  error: DangerIcon,
  folder: FolderIcon,
  help: QuestionMarkIcon,
  home: HomeIcon,
  info: InfoIcon,
  locationOn: PinIcon,
  lock: LockIcon,
  lockOpen: LockUnlockedIcon,
  mail: MailIcon,
  menu: MenuIcon,
  moreVert: OverflowIcon,
  moreHoriz: OverflowIcon,
  notifications: NotificationIcon,
  notificationsOff: NotificationOffIcon,
  pause: PauseIcon,
  person: UserIcon,
  photo: ImageIcon,
  play: PlayIcon,
  refresh: RefreshIcon,
  search: SearchIcon,
  send: SendIcon,
  settings: GearIcon,
  share: ShareIcon,
  star: StarIcon,
  starHalf: StarIcon,
  starOff: StarIcon,
  stop: StopIcon,
  upload: UploadIcon,
  visibility: VisibleIcon,
  visibilityOff: VisibleOffIcon,
  warning: WarningIcon,
  // DS-native aliases (handy for the agent / authors).
  calendar: CalendarIcon,
  danger: DangerIcon,
  gear: GearIcon,
  image: ImageIcon,
  list: ListIcon,
  pencil: PencilIcon,
  pin: PinIcon,
  plus: PlusIcon,
  question: QuestionMarkIcon,
  tag: TagIcon,
  trash: TrashIcon,
  user: UserIcon,
} satisfies Record<string, ComponentType>;

const DEFAULT_ICON: ComponentType = CircleOutlineIcon;

// `name` is an unconstrained DynamicString, so an inherited key like "constructor" would resolve
// to a non-component and crash the render. `Object.hasOwn` keeps those on the default-icon path.
const resolveDsIcon = (name: string): ComponentType =>
  Object.hasOwn(DS_ICON_BY_NAME, name) ? DS_ICON_BY_NAME[name as keyof typeof DS_ICON_BY_NAME] : DEFAULT_ICON;

/** Sorted list of supported icon names (used by the catalog schema + prompt). */
export const ICON_NAMES: string[] = Object.keys(DS_ICON_BY_NAME).sort();

/**
 * Schema (API) for the custom Icon component.
 */
const IconApi = {
  name: 'Icon',
  schema: z
    .object({
      name: DynamicStringSchema.describe(
        'The icon to display. Resolves to a Databricks Design System icon; unknown names render a neutral default.',
      ),
      size: z.number().describe('Icon size in pixels (defaults to the inherited font size).').optional(),
    })
    .strict(),
} satisfies ComponentApi;

export const Icon: ReactComponentImplementation = createComponentImplementation(IconApi, ({ props }) => {
  const name = typeof props.name === 'string' ? props.name : '';
  const IconComponent = resolveDsIcon(name);
  const size = typeof props.size === 'number' ? props.size : undefined;

  // DS icons size via font-size (they render at 1em), so set it on the wrapper.
  return (
    <span css={{ display: 'inline-flex', alignItems: 'center', ...(size ? { fontSize: size } : {}) }}>
      <IconComponent />
    </span>
  );
});
