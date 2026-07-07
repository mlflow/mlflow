/**
 * Copies text to the clipboard.
 *
 * Uses the modern Clipboard API when available (HTTPS / localhost).
 * Falls back to the legacy document.execCommand('copy') for plain-HTTP
 * deployments where navigator.clipboard is undefined.
 *
 * @returns Promise<boolean> — true if copy succeeded, false otherwise.
 */
export const copyToClipboard = async (text: string): Promise<boolean> => {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // fall through to execCommand
    }
  }

  // Fallback for insecure HTTP contexts
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.top = '0';
  textarea.style.left = '0';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  try {
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    document.body.removeChild(textarea);
  }
};
