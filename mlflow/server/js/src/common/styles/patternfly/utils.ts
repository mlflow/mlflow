export const convertRemStringToPx = (rem: string) => {
  return Number(rem.replace('rem', '')) * 16;
};

export const convertPxStringToPx = (px: string) => {
  return Number(px.replace('px', ''));
};
