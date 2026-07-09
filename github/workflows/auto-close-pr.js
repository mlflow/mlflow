// ignore the ModelsNextUIPromoModal modal window
const autoClosePr = async () => {
  try {
    // get the current page URL
    const currentPageUrl = await getCurrentPageUrl();
    if (currentPageUrl.includes('/#/models')) {
      // ignore the ModelsNextUIPromoModal modal window
      console.log('Ignoring ModelsNextUIPromoModal modal window');
      return;
    }
    // rest of the auto-close PR logic
  } catch (error) {
    console.error('Error auto-closing PR:', error);
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

autoClosePr();