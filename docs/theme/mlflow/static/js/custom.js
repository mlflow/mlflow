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

            setupSearch();
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

// feedback link
// var toField = "feedback@databricks.com";
// var subjectField = "Documentation Feedback";
// var bodyField = "I found an error on: " + document.URL;
// var feedbackUrlHref = "mailto:" + toField + "?subject=" + subjectField + "&body=" + encodeURI(bodyField)
// $('#feedbacklink').attr("href", feedbackUrlHref);


// managing embedded notebooks
$(".embedded-notebook").each(function() {
  var $container = $(this);
  var $iframe = $(this).find("iframe");

  $iframe.on("load", function() {
    var iframe = this;

    setTimeout(function() {
      // Set height
      var loadableFrame = iframe.contentWindow.document.getElementsByClassName("overallContainer")[0];
      var currentHeight = loadableFrame.scrollHeight;
      iframe.height = (currentHeight + 10) + "px";

      // Remove loading state
      $container.addClass("loaded");
    }, 4000);
  });

  // Load it
  $(this).waypoint({
    offset: "100%",
    handler: function() {
      this.destroy(); // turn off the Waypoint
      $iframe.attr("src", $iframe.data("src"));
    }
  });
});


// clipboard stuff
var languages = ["py", "scala", "sql", "r", "python"];

$('.code').map(function(i, val) {
    var $val = $(val)
    var lang = $val.attr("class").split(" ").filter(function (className) {
        return languages.indexOf(className) != -1;
    });

    var clippyStrStart = '<div class="clippy"><img src="' + CLIPPY_SVG_PATH + '" alt="Copy to clipboard">';
    var clippyStrEnd = '</div>';
    var copyStr = '<span>Copy</span>';
    var langStr = lang;


    $val.children(".highlight").prepend(clippyStrStart + copyStr + clippyStrEnd);
});

var clippy = new Clipboard('.clippy', {
    target: function(trigger) {
        return trigger.nextSibling;
    }
});

clippy.on("success", function(e) {
    var $clippy = $(e.trigger);
    $clippy.addClass("copied").find("span").text("Copied!");
    e.clearSelection();
    setTimeout(function(){
        $clippy.removeClass("copied").find("span").text("Copy");
    }, 1000);
});

function setupSearch() {
    docsearch({
      apiKey: algoliaConfigs.key,
      indexName: algoliaConfigs.index,
      inputSelector: '#algolia-search',
      debug: false,
      autocompleteOptions: {
        appendTo: '#algolia-wrapper',
        hint: false
      }
    });
    new Tether({
        element: '#algolia-wrapper',
        target: '#algolia-search',
        attachment: 'bottom left',
        targetAttachment: 'top left',
        targetOffset: '20px 0',
        constraints: [{
            to: 'window',
            attachment: 'together'
        }]
    });
}

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
