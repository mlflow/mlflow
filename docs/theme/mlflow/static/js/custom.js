require=(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({"sphinx-rtd-theme":[function(require,module,exports){ // NOLINT
    var jQuery = (typeof(window) != 'undefined') ? window.jQuery : require('jquery');

    // Sphinx theme nav state
    function ThemeNav () {

        var nav = {
            navBar: null,
            win: null,
            winScroll: false,
            winResize: false,
            linkScroll: false,
            winPosition: 0,
            winHeight: null,
            docHeight: null,
            isRunning: null
        };

        nav.enable = function () {
            var self = this;

            jQuery(function ($) {
                self.init($);

                self.reset();
                self.win.on('hashchange', self.reset);

                // Set scroll monitor
                self.win.on('scroll', function () {
                    if (!self.linkScroll) {
                        self.winScroll = true;
                    }
                });
                setInterval(function () { if (self.winScroll) self.onScroll(); }, 25);

                // Set resize monitor
                self.win.on('resize', function () {
                    self.winResize = true;
                });
                setInterval(function () { if (self.winResize) self.onResize(); }, 25);
                self.onResize();
            });
        };

        nav.init = function ($) {
            var doc = $(document),
                self = this;

            this.navBar = $('div.wy-side-scroll:first');
            this.win = $(window);

            // Set up javascript UX bits
            $(document)
            // Shift nav in mobile when clicking the menu.
                .on('click', "[data-toggle='wy-nav-top']", function() {
                    $("[data-toggle='wy-nav-shift']").toggleClass("shift");
                    $("[data-toggle='rst-versions']").toggleClass("shift");
                })

            // Nav menu link click operations
                .on('click', ".wy-menu-vertical .current ul li a", function() {
                    var target = $(this);
                    // Close menu when you click a link.
                    $("[data-toggle='wy-nav-shift']").removeClass("shift");
                    $("[data-toggle='rst-versions']").toggleClass("shift");
                    // Handle dynamic display of l3 and l4 nav lists
                    self.toggleCurrent(target);
                    self.hashChange();
                })
                .on('click', "[data-toggle='rst-current-version']", function() {
                    $("[data-toggle='rst-versions']").toggleClass("shift-up");
                })

            // Make tables responsive
            $("table.docutils:not(.field-list)")
                .wrap("<div class='wy-table-responsive'></div>");

            // Add expand links to all parents of nested ul
            $('.wy-menu-vertical ul').not('.simple').siblings('a').each(function () {
                var link = $(this);
                expand = $('<span class="toctree-expand"></span>');
                expand.on('click', function (ev) {
                    self.toggleCurrent(link);
                    ev.stopPropagation();
                    return false;
                });
                link.prepend(expand);
            });
        };

        nav.reset = function () {
            // Get anchor from URL and open up nested nav
            var anchor = encodeURI(window.location.hash);
            if (anchor) {
                try {
                    var link = $('.wy-menu-vertical')
                        .find('[href="' + anchor + '"]');
                    $('.wy-menu-vertical li.toctree-l1 li.current')
                        .removeClass('current');
                    link.closest('li.toctree-l2').addClass('current');
                    link.closest('li.toctree-l3').addClass('current');
                    link.closest('li.toctree-l4').addClass('current');
                }
                catch (err) {
                    console.log("Error expanding nav for anchor", err);
                }
            }
        };

        nav.onScroll = function () {
            this.winScroll = false;
            var newWinPosition = this.win.scrollTop(),
                winBottom = newWinPosition + this.winHeight,
                navPosition = this.navBar.scrollTop(),
                newNavPosition = navPosition + (newWinPosition - this.winPosition);
            if (newWinPosition < 0 || winBottom > this.docHeight) {
                return;
            }
            this.navBar.scrollTop(newNavPosition);
            this.winPosition = newWinPosition;
        };

        nav.onResize = function () {
            this.winResize = false;
            this.winHeight = this.win.height();
            this.docHeight = $(document).height();
        };

        nav.hashChange = function () {
            this.linkScroll = true;
            this.win.one('hashchange', function () {
                this.linkScroll = false;
            });
        };

        nav.toggleCurrent = function (elem) {
            var parent_li = elem.closest('li');
            parent_li.siblings('li.current').removeClass('current');
            parent_li.siblings().find('li.current').removeClass('current');
            parent_li.find('> ul li.current').removeClass('current');
            parent_li.toggleClass('current');
        }

        return nav;
    };

    module.exports.ThemeNav = ThemeNav();

    if (typeof(window) != 'undefined') {
        window.SphinxRtdTheme = { StickyNav: module.exports.ThemeNav };
    }

},{"jquery":"jquery"}]},{},["sphinx-rtd-theme"]);

// CUSTOM JS

// Affix the sidebar to the side if we scroll past the header,
// which is 55px. This ensures the sidebar is always visible,
// but makes room for the header if and only if the header is
// visible.
$(window).scroll(function() {
    var scrollTop = $(window).scrollTop();
    if (scrollTop <= 55) {
        $('.wy-nav-side').removeClass("fixed");
        $('.wy-nav-side').addClass("relative");
    } else {
        $('.wy-nav-side').addClass("fixed");
        $('.wy-nav-side').removeClass("relative");
    }
});

fetch('https://pypi.org/pypi/mlflow/json')
  .then((response) => response.json())
  .then((data) => {
    var versions = Object.keys(data.releases)
      // Drop dev/pre/rc/post versions and versions older than 1.0
      .filter(function (version) {
        return /^[1-9]+(\.\d+){0,3}$/.test(version);
      })
      // Sort versions
      // https://stackoverflow.com/a/40201629
      .map((a) =>
        a
          .split('.')
          .map((n) => +n + 100000)
          .join('.'),
      )
      .sort()
      .map((a) =>
        a
          .split('.')
          .map((n) => +n - 100000)
          .join('.'),
      )
      .reverse();

    var seenMinorVersions = [];
    var latestMicroVersions = [];
    versions.forEach(function (version) {
      var minor = version.split('.').slice(0, 2).join('.');
      if (!seenMinorVersions.includes(minor)) {
        seenMinorVersions.push(minor);
        latestMicroVersions.push(version);
      }
    });

    var latestVersion = latestMicroVersions[0];
    var docRegex = /\/docs\/(?<version>[^/]+)\//;
    var currentVersion = docRegex.exec(window.location.pathname).groups.version;
    var dropDown = document.createElement('select');
    dropDown.style = "margin-left: 5px";
    dropDown.onchange = function () {
      var newUrl = window.location.href.replace(docRegex, `/docs/${this.value}/`);
      window.location.assign(newUrl);
    };
    latestMicroVersions.forEach(function (version) {
      var option = document.createElement('option');
      option.value = version;
      option.selected = version === currentVersion;
      option.text = version === latestVersion ? `${version} (latest)` : version;
      dropDown.appendChild(option);
    });

    var versionTag = document.querySelector('span.version');
    versionTag.parentNode.replaceChild(dropDown, versionTag);
  })
  .catch((error) => {
    console.error('Failed to fetch package metadata from PyPI:', error);
  });

// Floating side bar navigable table of contents with current document h1, h2, h3 content listings
document.addEventListener("DOMContentLoaded", function() {

    // disable short-circuit for pages where a ToC doesn't make sense
    if (document.querySelector('.no-toc')) {
        debugLog('ToC disabled on this page.');
        return; 
    }

    var tocContainer = document.createElement('div');
    tocContainer.id = 'floating-toc-container';
    document.body.appendChild(tocContainer);

    var toc = document.createElement('ul');
    toc.id = 'floating-toc';
    tocContainer.appendChild(toc);

    var headings = document.querySelectorAll('[itemprop="articleBody"] h1, [itemprop="articleBody"] h2, [itemprop="articleBody"] h3');
    headings.forEach(function(heading, index) {
        if (!heading.id) {
            heading.id = 'heading-' + index;
        }
        var listItem = document.createElement('li');
        listItem.className = 'toc-' + heading.tagName.toLowerCase();
        listItem.innerHTML = '<a href="#' + heading.id + '">' + heading.textContent + '</a>';
        toc.appendChild(listItem);
    });

    // Stateful variable to store the ToC width after initial calculation
    var tocWidth;
    var retryCount = 0;
    var maxRetries = 5;
    var defaultToCWidth = 350; // Fallback width if calculation fails
    var isDebugMode = false; // Set this to true to enable debug logging

    function debugLog(...args) {
        if (isDebugMode) {
            console.log(...args);
        }
    }

    function checkToCPosition() {
        var tocContainer = document.getElementById('floating-toc-container');
        var contentElement = document.querySelector('.wy-nav-content');
        var sideScrollElement = document.querySelector('.wy-side-scroll');
    
        if (typeof tocWidth === 'undefined' || tocWidth <= 0) {
            debugLog('Calculating ToC width.');
            
            // Force the otherwise lazy creation of this DOM element to get sizing information
            tocContainer.style.visibility = 'hidden';
            tocContainer.style.position = 'absolute';
            tocContainer.style.display = 'block';
            tocWidth = tocContainer.offsetWidth;
            tocContainer.style.display = '';
            tocContainer.style.position = '';
            tocContainer.style.visibility = '';
            
            debugLog('ToC width set to:', tocWidth);
    
            if (tocWidth <= 0 && retryCount < maxRetries) {
                debugLog('ToC width is 0, scheduling a retry.');
                setTimeout(checkToCPosition, 100); 
                retryCount++; 
                return; 
            } else if (tocWidth <= 0) {
                debugLog('ToC width is 0 after retries, using default width.');
                tocWidth = defaultToCWidth;
            }
        }
    
        var contentRect = contentElement.getBoundingClientRect();
        var sideScrollRect = sideScrollElement.getBoundingClientRect();
        var availableSpace = window.innerWidth - contentRect.width - sideScrollRect.right - tocWidth;

        debugLog('Window width:', window.innerWidth);
        debugLog('Content rect:', contentRect);
        debugLog('Side scroll rect:', sideScrollRect);
        debugLog('ToC width:', tocWidth);
        debugLog('Available space:', availableSpace);
    
        var shouldDisplay = availableSpace >= 0; 
        tocContainer.style.display = shouldDisplay ? 'block' : 'none';
    
        debugLog('ToC should display:', shouldDisplay);
    }
    
    window.addEventListener('resize', checkToCPosition);
    debugLog('Resize observer set up.');

    checkToCPosition();
    debugLog('DOMContentLoaded - Initial ToC check performed.');
});

// Scroll highlight functionality for the floating ToC
document.addEventListener('DOMContentLoaded', (event) => {
    const headings = document.querySelectorAll('[itemprop="articleBody"] h1, [itemprop="articleBody"] h2, [itemprop="articleBody"] h3');
    const navLinks = document.querySelectorAll('#floating-toc li a');

    function onScroll() {
        let currentSectionId = '';

        // Use the bounding rectangle and scroll position to determine the current section
        headings.forEach((heading) => {
            const rect = heading.getBoundingClientRect();
            if (rect.top <= window.innerHeight * 0.25) { 
                currentSectionId = heading.id;
            }
        });

        // Update the ToC links' active state based on the current section
        navLinks.forEach((link) => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + currentSectionId) {
                link.classList.add('active');
            }
        });
    }

    window.addEventListener('scroll', onScroll);
    onScroll();
});


