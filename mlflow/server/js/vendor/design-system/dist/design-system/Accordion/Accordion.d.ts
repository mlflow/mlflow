import type { Interpolation, Theme as EmotionTheme } from '@emotion/react';
import type { CollapseProps as AntDCollapseProps, CollapsePanelProps as AntDCollapsePanelProps } from 'antd';
import type { CollapsibleType } from 'antd/lib/collapse/CollapsePanel';
import { DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';
import type { AnalyticsEventValueChangeNoPiiFlagProps, HTMLDataAttributes } from '../types';
export interface AccordionProps extends HTMLDataAttributes, AnalyticsEventValueChangeNoPiiFlagProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
    /** How many sections can be displayed at once */
    displayMode?: 'single' | 'multiple';
    /** Key of the active panel */
    activeKey?: Array<string | number> | string | number;
    /** Specify whether the entire header (`undefined` (default)) or children (`"header"`) are the collapsible trigger. `"disabled"` disables the collapsible behavior of the accordion headers. */
    collapsible?: CollapsibleType;
    /** Key of the initial active panel */
    defaultActiveKey?: Array<string | number> | string | number;
    /** Destroy Inactive Panel */
    destroyInactivePanel?: boolean;
    /** Callback function executed when active panel is changed */
    onChange?: (key: string | string[]) => void;
    alignContentToEdge?: boolean;
    /** Escape hatch to allow passing props directly to the underlying Ant `TabPane` component. */
    dangerouslySetAntdProps?: Partial<AntDCollapseProps>;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
    chevronAlignment?: 'left' | 'right';
}
export interface AccordionPanelProps extends HTMLDataAttributes {
    children: React.ReactNode;
    /** Unique key identifying the panel from among its siblings */
    key: string | number;
    /** Title of the panel */
    header: React.ReactNode;
    /** Specify whether the entire header (`undefined` (default)) or children (`"header"`) are the collapsible trigger. `"disabled"` disables the collapsible behavior of the accordion headers. */
    collapsible?: CollapsibleType;
    /** Forced render of content on panel, instead of lazy rending after clicking on header */
    forceRender?: boolean;
    /** Escape hatch to allow passing props directly to the underlying Ant `TabPane` component. */
    dangerouslySetAntdProps?: Partial<AntDCollapsePanelProps>;
    /** Applies emotion styles to the top-level element in the component. Ask in #dubois before using. */
    dangerouslyAppendEmotionCSS?: Interpolation<EmotionTheme>;
}
interface AccordionInterface extends React.FC<AccordionProps> {
    Panel: React.FC<AccordionPanelProps>;
}
export declare const AccordionPanel: React.FC<AccordionPanelProps>;
export declare const Accordion: AccordionInterface;
export {};
//# sourceMappingURL=Accordion.d.ts.map