
$(function() {

  $('div.code-section').each(function() {
    var example_sel = $('<ul />', { class: 'section-selector' });
    var i = 0;
    $('div[class^="highlight-"]', this).each(function() {
      language_name = $(this).attr('class').substring(10).replace('notranslate', '');
      language_name = language_name.charAt(0).toUpperCase() + language_name.substr(1);

      var sel_item = $('<li />', {
          class: $(this).attr('class'),
          text: language_name
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

  $('div.plain-section').each(function() {
    var example_sel = $('<ul />', { class: 'section-selector' });
    var i = 0;
    $('div.container', this).each(function() {
      var language_name = $(this).attr('class').replace(' docutils container', '').trim();
      language_name = language_name.charAt(0).toUpperCase() + language_name.substr(1);

      var sel_item = $('<li />', {
          class: $(this).attr('class'),
          text: language_name
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

  $('div.code-section ul.section-selector li,div.plain-section ul.section-selector li').click(function(evt) {
    evt.preventDefault();

    var sel_class = $(this).attr('class')
      .replace(' docutils container', '')
      .replace('notranslate', '')
      .replace(' selected', '');

    $('ul.section-selector li').each(function() {
      var parent = $(this).parent().parent();
      var my_sel_class = sel_class;
      // When the target language is not available, default to bash or python.
      if (!$('div.' + sel_class, parent).length) {
        if ($('div.highlight-bash', parent).length)
          my_sel_class = 'highlight-bash';
        else
          my_sel_class = 'highlight-python';
      }

      $('div.example', parent).hide();
      $('div.' + my_sel_class, parent).show();

      $('ul.section-selector li', parent).removeClass('selected');
      $('ul.section-selector li.' + my_sel_class, parent).addClass('selected');
    });
  });

});

