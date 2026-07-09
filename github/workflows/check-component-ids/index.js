// ignore the ModelsNextUIPromoModal modal window
const checkComponentIds = async () => {
  try {
    // get the current page URL
    const currentPageUrl = await getCurrentPageUrl();
    if (currentPageUrl.includes('/#/models')) {
      // ignore the ModelsNextUIPromoModal modal window
      console.log('Ignoring ModelsNextUIPromoModal modal window');
      return;
    }
    // rest of the check-component-ids logic
  } catch (error) {
    console.error('Error checking component IDs:', error);
  }
};

// helper function to get the current page URL
const getCurrentPageUrl = async () => {
  try {
    const response = await fetch('/api/current-page-url');
    return response.json();
  } catch (error) {
    console.error('Error getting current page URL:', error);
    return null;
  }
};

checkComponentIds();