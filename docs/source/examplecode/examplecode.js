/**
 * Dynamic multiple language example code block.
 */

$(function() {

  $('div.example-code').each(function() {
    var example_sel = $('<ul />', { class: "example-selector" });
    var i = 0;
    $('div[class^="highlight-"]', this).each(function() {
      var sel_item = $('<li />', {
          class: $(this).attr('class'),
          text: $(this).attr('class').substring(10)
      });
      if (i++) {
        $(this).hide();
      } else {
        sel_item.addClass('selected');
      }
      example_sel.append(sel_item);
      $(this).addClass('example');
    });
    $(this).prepend(example_sel);
    example_sel = null;
    i = null;
  });

  $('div.example-code ul.example-selector li').click(function(evt) {
    evt.preventDefault();
    $('ul.example-selector li').removeClass('selected');
    var sel_class = $(this).attr('class');
    $('div.example').hide();
    $('div.' + sel_class).show();
    $('ul.example-selector li.' + sel_class).addClass('selected');
    sel_class = null;
  });

});

