/// <reference types="react" />
type MessageHeaderProps = {
    userName: React.ReactNode;
    customHeaderIcon?: React.ReactNode;
    avatarURL?: string;
    leftContent?: React.ReactNode;
    rightContent?: React.ReactNode;
};
declare const MessageHeader: ({ userName, customHeaderIcon, avatarURL, leftContent, rightContent }: MessageHeaderProps) => import("@emotion/react/jsx-runtime").JSX.Element;
export default MessageHeader;
//# sourceMappingURL=MessageHeader.d.ts.map