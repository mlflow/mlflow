import React from 'react';
import type { DangerousGeneralProps, HTMLDataAttributes } from '../types';
export interface EmptyProps extends HTMLDataAttributes, DangerousGeneralProps {
    image?: JSX.Element;
    title?: React.ReactNode;
    description: React.ReactNode;
    button?: React.ReactNode;
}
export declare const Empty: React.FC<EmptyProps>;
//# sourceMappingURL=Empty.d.ts.map