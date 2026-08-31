/* Healix chat client.
   One AJAX round trip per question; the transcript is rebuilt from the JSON the
   API returns so the DOM never holds state the server doesn't know about. */

$(function () {
  var $transcript = $('#transcript');
  var $question = $('#question');
  var $send = $('#send');
  var $empty = $('#empty-state');
  var pending = false;

  function escapeHtml(value) {
    return $('<div>').text(value == null ? '' : value).html();
  }

  function autoGrow() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 128) + 'px';
  }

  function scrollToEnd() {
    $('html, body').animate({ scrollTop: $(document).height() }, 200);
  }

  function renderCitations(sources) {
    if (!sources || !sources.length) return '';
    var rows = sources.map(function (source) {
      var page = source.page ? 'p. ' + source.page : '—';
      return '<li class="citation">' +
             '<span class="citation-page">' + escapeHtml(page) + '</span>' +
             '<span class="citation-snippet">' + escapeHtml(source.snippet) + '</span>' +
             '</li>';
    }).join('');
    return '<ul class="citations">' + rows + '</ul>';
  }

  function appendTurn(question) {
    $empty.remove();
    var id = 'turn-' + Date.now();
    $transcript.append(
      '<article class="turn" id="' + id + '">' +
        '<p class="turn-question">' + escapeHtml(question) + '</p>' +
        '<div class="thinking" role="status" aria-label="Searching the encyclopedia">' +
          '<span></span><span></span><span></span>' +
        '</div>' +
      '</article>'
    );
    scrollToEnd();
    return $('#' + id);
  }

  function fillAnswer($turn, data) {
    var timing = 'retrieved in ' + data.retrieval_ms + ' ms · answered in ' +
                 (data.total_ms / 1000).toFixed(1) + ' s';
    $turn.find('.thinking').replaceWith(
      '<p class="turn-answer">' + escapeHtml(data.answer) + '</p>' +
      renderCitations(data.sources) +
      '<p class="timing">' + timing + '</p>'
    );
    scrollToEnd();
  }

  function fillError($turn, message) {
    $turn.find('.thinking').replaceWith(
      '<p class="turn-answer is-error">' + escapeHtml(message) + '</p>'
    );
  }

  function setPending(state) {
    pending = state;
    $send.prop('disabled', state).text(state ? 'Asking…' : 'Ask');
  }

  function submit() {
    var question = ($question.val() || '').trim();
    if (!question || pending) return;

    var $turn = appendTurn(question);
    $question.val('').trigger('input');
    setPending(true);

    $.ajax({
      url: '/api/chat',
      method: 'POST',
      contentType: 'application/json',
      dataType: 'json',
      data: JSON.stringify({ question: question })
    })
      .done(function (data) { fillAnswer($turn, data); })
      .fail(function (xhr) {
        var message = 'Healix could not reach the retrieval service. Try again in a moment.';
        if (xhr.responseJSON && xhr.responseJSON.error) message = xhr.responseJSON.error;
        fillError($turn, message);
      })
      .always(function () {
        setPending(false);
        $question.trigger('focus');
      });
  }

  $question.on('input', autoGrow);

  $question.on('keydown', function (event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  });

  $send.on('click', submit);

  $transcript.on('click', '.starter', function () {
    $question.val($(this).text());
    submit();
  });

  $question.trigger('focus');
});
