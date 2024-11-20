export function postProcessSidebar(items) {
  // Remove items with customProps.hide set to true
  return items.filter((item) => item.customProps?.hide !== true);
}
