export interface RegistryIconImage {
  src: string;
  sizes?: string[];
  mimeType?: string;
  theme?: string;
}

export const resolveIcon = <T extends RegistryIconImage>(icons?: T[] | null, isDarkMode?: boolean): T | undefined => {
  if (!icons?.length) return undefined;
  const preferred = isDarkMode ? 'dark' : 'light';
  return icons.find((icon) => icon.theme === preferred) ?? icons.find((icon) => !icon.theme);
};

export const sanitizeHref = (url: string | undefined): string | undefined => {
  if (!url) return undefined;
  try {
    const parsed = new URL(url);
    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return url;
    }
  } catch {
    // malformed URL
  }
  return undefined;
};
