export const toRGBA = (color: string, alpha: number): string => {
  // Helper function to parse RGB values
  const parseRGB = (rgb: string): number[] => {
    const matches = rgb.match(/\d+/g);
    return matches ? matches.map(Number).concat([0, 0]).slice(0, 3) : [0, 0, 0];
  };

  // Helper function to convert hex to RGB
  const hexToRGB = (hex: string): number[] => {
    const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    const normalizedHex = hex.replace(shorthandRegex, (_, r, g, b) => r + r + g + g + b + b);
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(normalizedHex);
    return result ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)] : [0, 0, 0];
  };

  // Helper function to convert named colors to RGB
  const namedColorToRGB = (name: string): number[] => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return [0, 0, 0];
    ctx.fillStyle = name;
    return parseRGB(ctx.fillStyle);
  };

  let rgb: number[];

  if (color.startsWith('rgb')) {
    rgb = parseRGB(color);
  } else if (color.startsWith('#')) {
    rgb = hexToRGB(color);
  } else {
    // For named colors and other formats
    rgb = namedColorToRGB(color);
  }

  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
};
