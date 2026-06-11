import type { ComponentType } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
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
// native to the rest of MLflow. Keys cover the basic A2UI catalog's
// Material-style names (so an agent trained on those still resolves), plus a
// few DS-native aliases. Names with no DS equivalent fall back to a neutral
// default icon rather than the Material Symbols font (which MLflow doesn't
// load) — see DEFAULT_ICON below.
const DS_ICON_BY_NAME: Record<string, ComponentType> = {
  // Basic-catalog (Material-style) names → closest DS icon.
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
};

const DEFAULT_ICON: ComponentType = CircleOutlineIcon;

/** Sorted list of supported icon names (used by the catalog schema + prompt). */
export const ICON_NAMES = Object.keys(DS_ICON_BY_NAME).sort();

/**
 * Schema (API) for the custom Icon component. Renders a single Databricks
 * Design System icon by name. Unknown names degrade gracefully to a neutral
 * default icon, so the agent can request any name without breaking the layout.
 */
export const IconApi = {
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

export const Icon = createComponentImplementation(IconApi, ({ props }) => {
  const name = typeof props.name === 'string' ? props.name : '';
  const IconComponent = DS_ICON_BY_NAME[name] ?? DEFAULT_ICON;
  const size = typeof props.size === 'number' ? props.size : undefined;

  // DS icons size via font-size (they render at 1em), so set it on the wrapper.
  return (
    <span css={{ display: 'inline-flex', alignItems: 'center', ...(size ? { fontSize: size } : {}) }}>
      <IconComponent />
    </span>
  );
});
