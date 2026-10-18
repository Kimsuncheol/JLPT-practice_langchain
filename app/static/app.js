// Ephemeral browser-memory state ONLY (never persisted to localStorage, sessionStorage, IndexedDB, cookies, or database)
const MAX_RECENT_MESSAGES = 6;
let currentPracticeSet = null;
let selectedAnswers = {};
let isQuizSubmitted = false;
let activeTutorQuestionIndex = 0;
let conversationHistory = [];
let currentWordExplanation = null;
let sessionSummary = {
  setsCompleted: 0,
  answeredCount: 0,
  correctCount: 0,
  missedTopics: []
};

function setStatusBanner(elementId, message) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (!message) {
    el.style.display = 'none';
    el.textContent = '';
  } else {
    el.style.display = 'block';
    el.textContent = message;
  }
}

function setErrorBanner(elementId, message) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (!message) {
    el.style.display = 'none';
    el.textContent = '';
  } else {
    el.style.display = 'block';
    el.textContent = message;
  }
}

function syncQuestionCountLimit() {
  const cat = document.getElementById('category').value;
  const countSelect = document.getElementById('question_count');
  const maxAllowed = cat === 'reading' ? 3 : 5;
  Array.from(countSelect.options).forEach((opt) => {
    const val = parseInt(opt.value, 10);
    opt.disabled = val > maxAllowed;
  });
  if (parseInt(countSelect.value, 10) > maxAllowed) {
    countSelect.value = String(maxAllowed);
  }
}

function updateTutorQuestionSelector() {
  const sel = document.getElementById('tutor-question-select');
  if (!sel) return;
  sel.innerHTML = '';
  if (!currentPracticeSet || !currentPracticeSet.questions || currentPracticeSet.questions.length === 0) {
    const opt = document.createElement('option');
    opt.value = '0';
    opt.textContent = 'No active question (generate a practice set first)';
    sel.appendChild(opt);
    return;
  }
  currentPracticeSet.questions.forEach((q, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = `Q${idx + 1}: ${q.question_text}`;
    opt.selected = idx === activeTutorQuestionIndex;
    sel.appendChild(opt);
  });
}

function selectTutorQuestion(indexVal) {
  activeTutorQuestionIndex = parseInt(indexVal, 10) || 0;
  const sel = document.getElementById('tutor-question-select');
  if (sel) sel.value = String(activeTutorQuestionIndex);
}

async function generatePracticeSet() {
  syncQuestionCountLimit();
  const level = document.getElementById('level').value;
  const category = document.getElementById('category').value;
  const question_count = parseInt(document.getElementById('question_count').value, 10);
  const explanation_language = document.getElementById('explanation_language').value;
  const genBtn = document.getElementById('generate-btn');

  currentPracticeSet = null;
  selectedAnswers = {};
  isQuizSubmitted = false;
  activeTutorQuestionIndex = 0;
  document.getElementById('quiz-container').innerHTML = '';
  document.getElementById('quiz-score-summary').innerHTML = '';
  document.getElementById('quiz-actions').style.display = 'none';
  setErrorBanner('quiz-error', '');
  setStatusBanner(
    'quiz-status',
    'Loading: Generating unofficial AI practice set (first request after idle may take 30–60s during a Render Free cold start)...'
  );
  if (genBtn) {
    genBtn.disabled = true;
    genBtn.setAttribute('aria-busy', 'true');
  }

  try {
    const res = await fetch('/api/practice/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify({ level, category, question_count, explanation_language })
    });
    const data = await res.json();
    if (!res.ok) {
      setStatusBanner('quiz-status', '');
      setErrorBanner('quiz-error', data.detail || 'Failed to generate practice set.');
      return;
    }
    currentPracticeSet = data;
    setStatusBanner(
      'quiz-status',
      `${data.content_label} (${data.difficulty_label}) — Select an answer for each question, then click Submit Answers.`
    );
    updateTutorQuestionSelector();
    renderQuiz();
  } catch (err) {
    setStatusBanner('quiz-status', '');
    setErrorBanner('quiz-error', 'Network error while generating practice set.');
  } finally {
    if (genBtn) {
      genBtn.disabled = false;
      genBtn.removeAttribute('aria-busy');
    }
  }
}

function selectOption(questionId, optionIndex) {
  if (isQuizSubmitted) return;
  selectedAnswers[questionId] = optionIndex;
  setErrorBanner('quiz-error', '');
}

function compareAnswerLocally(question, selectedIndex) {
  const isCorrect = selectedIndex === question.correct_index;
  const correctOptionText = question.correct_answer;
  const whyCorrectFits = question.option_explanations[question.correct_index];
  const whyAlternativesDoNotFit = [];

  question.options.forEach((optText, idx) => {
    if (idx !== question.correct_index) {
      whyAlternativesDoNotFit.push({
        optionIndex: idx,
        optionText: optText,
        wasSelected: idx === selectedIndex,
        explanation: question.option_explanations[idx]
      });
    }
  });

  return {
    isCorrect,
    selectedIndex,
    selectedOptionText: typeof selectedIndex === 'number' ? question.options[selectedIndex] : null,
    correctIndex: question.correct_index,
    correctOptionText,
    supportingSentence: question.supporting_sentence || null,
    whyCorrectFits,
    whyAlternativesDoNotFit
  };
}

function submitQuizLocally() {
  if (!currentPracticeSet || !currentPracticeSet.questions || isQuizSubmitted) return;
  const unanswered = currentPracticeSet.questions.filter(
    (q) => typeof selectedAnswers[q.id] !== 'number'
  );
  if (unanswered.length > 0) {
    setErrorBanner(
      'quiz-error',
      `Please select an option for all ${currentPracticeSet.questions.length} question(s) before submitting.`
    );
    return;
  }

  setErrorBanner('quiz-error', '');
  // Local grading in browser memory without calling Ollama or any backend endpoint
  isQuizSubmitted = true;

  // Update in-memory session summary (accuracy and missed topics)
  sessionSummary.setsCompleted += 1;
  currentPracticeSet.questions.forEach((q) => {
    const evalResult = compareAnswerLocally(q, selectedAnswers[q.id]);
    sessionSummary.answeredCount += 1;
    if (evalResult.isCorrect) {
      sessionSummary.correctCount += 1;
    } else {
      sessionSummary.missedTopics.push({
        level: currentPracticeSet.level,
        category: currentPracticeSet.category,
        questionText: q.question_text,
        selectedAnswer: evalResult.selectedOptionText || '',
        correctAnswer: evalResult.correctOptionText
      });
    }
  });

  renderQuiz();
  renderSessionSummary();
}

function renderSessionSummary() {
  const accEl = document.getElementById('session-accuracy-value');
  const countEl = document.getElementById('session-count-value');
  const missedContainer = document.getElementById('session-missed-topics');
  if (!accEl || !countEl || !missedContainer) return;

  if (sessionSummary.answeredCount === 0) {
    accEl.textContent = '—';
    countEl.textContent = '0 / 0';
  } else {
    const pct = Math.round((sessionSummary.correctCount / sessionSummary.answeredCount) * 100);
    accEl.textContent = `${pct}%`;
    countEl.textContent = `${sessionSummary.correctCount} / ${sessionSummary.answeredCount}`;
  }

  missedContainer.innerHTML = '';
  if (sessionSummary.missedTopics.length === 0) {
    const emptyMsg = document.createElement('p');
    emptyMsg.style.margin = '0';
    emptyMsg.style.fontSize = '0.9rem';
    emptyMsg.style.color = '#475569';
    emptyMsg.textContent = 'No missed questions in this session.';
    missedContainer.appendChild(emptyMsg);
    return;
  }

  const ul = document.createElement('ul');
  ul.className = 'missed-list';
  sessionSummary.missedTopics.forEach((item) => {
    const li = document.createElement('li');
    const topicBadge = document.createElement('strong');
    topicBadge.textContent = `[${item.level} · ${item.category}] `;
    li.appendChild(topicBadge);

    const qSpan = document.createElement('span');
    qSpan.setAttribute('lang', 'ja');
    qSpan.className = 'jp-text';
    qSpan.textContent = item.questionText;
    li.appendChild(qSpan);

    const detailSpan = document.createElement('span');
    detailSpan.textContent = ` (Chosen: ${item.selectedAnswer} → Correct: ${item.correctAnswer})`;
    li.appendChild(detailSpan);

    ul.appendChild(li);
  });
  missedContainer.appendChild(ul);
}

function renderQuiz() {
  const container = document.getElementById('quiz-container');
  const actionsEl = document.getElementById('quiz-actions');
  const summaryEl = document.getElementById('quiz-score-summary');
  container.innerHTML = '';
  summaryEl.innerHTML = '';

  if (!currentPracticeSet || !currentPracticeSet.questions) {
    actionsEl.style.display = 'none';
    return;
  }

  if (currentPracticeSet.passage) {
    const passageBox = document.createElement('div');
    passageBox.className = 'passage-card';

    const pTitle = document.createElement('div');
    const titleStrong = document.createElement('strong');
    titleStrong.textContent = 'Reading Passage ';
    pTitle.appendChild(titleStrong);

    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = 'Unofficial AI Reading (Approx. Difficulty)';
    pTitle.appendChild(badge);
    passageBox.appendChild(pTitle);

    const pBody = document.createElement('p');
    pBody.setAttribute('lang', 'ja');
    pBody.className = 'jp-text';
    pBody.textContent = currentPracticeSet.passage;
    passageBox.appendChild(pBody);

    container.appendChild(passageBox);
  }

  let correctCount = 0;

  currentPracticeSet.questions.forEach((q, qIdx) => {
    const card = document.createElement('div');
    card.className = 'question-card';

    const fieldset = document.createElement('fieldset');
    fieldset.className = 'question-fieldset';

    const legend = document.createElement('legend');
    legend.className = 'question-legend';

    const leftTitle = document.createElement('span');
    const qNum = document.createElement('strong');
    qNum.textContent = `Q${qIdx + 1}. `;
    leftTitle.appendChild(qNum);

    const qText = document.createElement('span');
    qText.setAttribute('lang', 'ja');
    qText.className = 'jp-text';
    qText.textContent = q.question_text;
    leftTitle.appendChild(qText);
    legend.appendChild(leftTitle);

    const rightControls = document.createElement('span');
    const aiTag = document.createElement('span');
    aiTag.className = 'badge';
    aiTag.textContent = 'Unofficial AI (Approx. Level)';
    rightControls.appendChild(aiTag);

    const askBtn = document.createElement('button');
    askBtn.type = 'button';
    askBtn.className = 'secondary';
    askBtn.style.marginLeft = '0.5rem';
    askBtn.style.fontSize = '0.8rem';
    askBtn.style.padding = '0.2rem 0.55rem';
    askBtn.textContent = 'Focus in Tutor';
    askBtn.onclick = () => {
      selectTutorQuestion(qIdx);
      const chatInput = document.getElementById('chat-input');
      if (chatInput) chatInput.focus();
    };
    rightControls.appendChild(askBtn);
    legend.appendChild(rightControls);

    fieldset.appendChild(legend);

    q.options.forEach((optText, optIdx) => {
      const label = document.createElement('label');
      label.className = 'option-label';
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = `question-${q.id}`;
      radio.value = String(optIdx);
      radio.checked = selectedAnswers[q.id] === optIdx;
      radio.disabled = isQuizSubmitted;
      radio.onchange = () => selectOption(q.id, optIdx);

      const optSpan = document.createElement('span');
      optSpan.setAttribute('lang', 'ja');
      optSpan.className = 'jp-text';
      optSpan.textContent = `${optIdx + 1}. ${optText}`;

      label.appendChild(radio);
      label.appendChild(optSpan);
      fieldset.appendChild(label);
    });

    card.appendChild(fieldset);

    // Reveal explanations ONLY after submission (rendered safely with textContent)
    if (isQuizSubmitted) {
      const evalResult = compareAnswerLocally(q, selectedAnswers[q.id]);
      if (evalResult.isCorrect) correctCount += 1;

      const feedbackPanel = document.createElement('div');
      feedbackPanel.className = 'feedback-panel';

      const statusBanner = document.createElement('div');
      statusBanner.className = evalResult.isCorrect ? 'correct-banner' : 'incorrect-banner';
      statusBanner.textContent = evalResult.isCorrect
        ? '✓ Correct'
        : `✗ Incorrect (You selected: ${evalResult.selectedOptionText})`;
      feedbackPanel.appendChild(statusBanner);

      const correctLine = document.createElement('p');
      const correctStrong = document.createElement('strong');
      correctStrong.textContent = 'Correct Answer: ';
      correctLine.appendChild(correctStrong);
      correctLine.appendChild(
        document.createTextNode(`${evalResult.correctIndex + 1}. ${evalResult.correctOptionText}`)
      );
      feedbackPanel.appendChild(correctLine);

      if (evalResult.supportingSentence) {
        const supportLine = document.createElement('p');
        const supportStrong = document.createElement('strong');
        supportStrong.textContent = 'Supporting Sentence from Passage: ';
        supportLine.appendChild(supportStrong);
        const supportText = document.createElement('span');
        supportText.setAttribute('lang', 'ja');
        supportText.className = 'jp-text';
        supportText.textContent = evalResult.supportingSentence;
        supportLine.appendChild(supportText);
        feedbackPanel.appendChild(supportLine);
      }

      const whyFitsLine = document.createElement('p');
      const whyFitsStrong = document.createElement('strong');
      whyFitsStrong.textContent = 'Why it fits: ';
      whyFitsLine.appendChild(whyFitsStrong);
      whyFitsLine.appendChild(document.createTextNode(evalResult.whyCorrectFits));
      feedbackPanel.appendChild(whyFitsLine);

      const altHeader = document.createElement('p');
      const altStrong = document.createElement('strong');
      altStrong.textContent = 'Why the alternatives do not fit:';
      altHeader.appendChild(altStrong);
      feedbackPanel.appendChild(altHeader);

      const altList = document.createElement('ul');
      evalResult.whyAlternativesDoNotFit.forEach((alt) => {
        const li = document.createElement('li');
        const prefix = alt.wasSelected ? '[Your choice] ' : '';
        li.textContent = `${prefix}${alt.optionIndex + 1}. ${alt.optionText} — ${alt.explanation}`;
        altList.appendChild(li);
      });
      feedbackPanel.appendChild(altList);

      const followUpRow = document.createElement('div');
      followUpRow.className = 'row follow-up-actions';
      followUpRow.setAttribute('role', 'group');
      followUpRow.setAttribute('aria-label', 'Follow-up question options');
      followUpRow.style.marginTop = '0.75rem';
      followUpRow.style.marginBottom = '0';

      const againBtn = document.createElement('button');
      againBtn.type = 'button';
      againBtn.className = 'secondary follow-up-btn';
      againBtn.textContent = 'Practice this again';
      againBtn.onclick = () => generateFollowUpQuestion(qIdx, 'same');
      followUpRow.appendChild(againBtn);

      const harderBtn = document.createElement('button');
      harderBtn.type = 'button';
      harderBtn.className = 'follow-up-btn';
      harderBtn.textContent = 'Try a harder question';
      harderBtn.onclick = () => generateFollowUpQuestion(qIdx, 'harder');
      followUpRow.appendChild(harderBtn);

      feedbackPanel.appendChild(followUpRow);

      card.appendChild(feedbackPanel);
    }

    container.appendChild(card);
  });

  actionsEl.style.display = isQuizSubmitted ? 'none' : 'flex';

  if (isQuizSubmitted) {
    const total = currentPracticeSet.questions.length;
    const scoreBox = document.createElement('div');
    scoreBox.className = 'notice';
    scoreBox.textContent =
      `Batch Score: ${correctCount} / ${total} — Unofficial AI practice (approximate difficulty). Graded locally in browser memory against the inspectable answer key (not tamper-proof, not stored, and does not predict an official JLPT score).`;
    summaryEl.appendChild(scoreBox);
  }
}

async function generateFollowUpQuestion(questionIndex, requestedDifficulty) {
  if (
    !currentPracticeSet ||
    !currentPracticeSet.questions ||
    !currentPracticeSet.questions[questionIndex]
  ) {
    return;
  }

  const currentQuestion = currentPracticeSet.questions[questionIndex];
  const selIdx = selectedAnswers[currentQuestion.id];
  const selectedAnswerText =
    typeof selIdx === 'number' && currentQuestion.options[selIdx]
      ? currentQuestion.options[selIdx]
      : currentQuestion.options[0];

  const level = currentPracticeSet.level || document.getElementById('level').value;
  const category = currentPracticeSet.category || document.getElementById('category').value;
  const explanation_language =
    currentPracticeSet.explanation_language ||
    document.getElementById('explanation_language').value;

  const followUpBtns = document.querySelectorAll('.follow-up-btn');
  followUpBtns.forEach((btn) => {
    btn.disabled = true;
    btn.setAttribute('aria-busy', 'true');
  });

  setErrorBanner('quiz-error', '');
  const modeLabel =
    requestedDifficulty === 'harder' ? 'Try a harder question' : 'Practice this again';
  setStatusBanner(
    'quiz-status',
    `Loading: Generating follow-up question (${modeLabel}) targeting the same learning point...`
  );

  try {
    const res = await fetch('/api/practice/follow-up', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify({
        level,
        category,
        explanation_language,
        difficulty: requestedDifficulty,
        selected_answer: selectedAnswerText,
        current_item: {
          id: currentQuestion.id,
          passage: currentPracticeSet.passage || null,
          question_text: currentQuestion.question_text,
          options: currentQuestion.options,
          correct_index: currentQuestion.correct_index,
          correct_answer: currentQuestion.correct_answer,
          selected_answer: selectedAnswerText,
          supporting_sentence: currentQuestion.supporting_sentence || null,
          option_explanations: currentQuestion.option_explanations
        }
      })
    });
    const data = await res.json();
    if (!res.ok) {
      setStatusBanner('quiz-status', '');
      setErrorBanner('quiz-error', data.detail || 'Failed to generate follow-up question.');
      return;
    }

    const followUpQuestion = data.question || {
      id: data.id,
      question_text: data.question_text,
      options: data.options,
      correct_index: data.correct_index,
      correct_answer: data.correct_answer,
      option_explanations: data.option_explanations,
      supporting_sentence: data.supporting_sentence || null,
      ai_generated: true,
      difficulty_label: data.difficulty_label,
      content_label: data.content_label
    };

    // Replace active practice set in browser memory and keep solution hidden until submission
    currentPracticeSet = {
      level: data.level || level,
      category: data.category || category,
      question_count: 1,
      explanation_language: data.explanation_language || explanation_language,
      passage: data.passage || null,
      ai_generated: true,
      unofficial_practice: true,
      difficulty_label: data.difficulty_label,
      content_label: data.content_label,
      questions: [followUpQuestion]
    };
    selectedAnswers = {};
    isQuizSubmitted = false;
    activeTutorQuestionIndex = 0;

    setStatusBanner(
      'quiz-status',
      `Follow-up question ready (${modeLabel}) — Select an answer and click Submit Answers to reveal the explanation.`
    );
    updateTutorQuestionSelector();
    renderQuiz();
  } catch (err) {
    setStatusBanner('quiz-status', '');
    setErrorBanner('quiz-error', 'Network error while generating follow-up question.');
  } finally {
    followUpBtns.forEach((btn) => {
      btn.disabled = false;
      btn.removeAttribute('aria-busy');
    });
  }
}

function getSelectedQuestionPayload() {
  if (
    currentPracticeSet &&
    currentPracticeSet.questions &&
    currentPracticeSet.questions[activeTutorQuestionIndex]
  ) {
    const q = currentPracticeSet.questions[activeTutorQuestionIndex];
    const selIdx = selectedAnswers[q.id];
    return {
      id: q.id,
      passage: currentPracticeSet.passage || null,
      question_text: q.question_text,
      options: q.options,
      correct_index: q.correct_index,
      correct_answer: q.correct_answer,
      selected_answer: typeof selIdx === 'number' ? q.options[selIdx] : null,
      supporting_sentence: q.supporting_sentence || null,
      option_explanations: q.option_explanations
    };
  }
  return {
    question_text: 'General JLPT practice item inquiry'
  };
}

function renderChat() {
  const box = document.getElementById('chat-box');
  if (!box) return;
  box.innerHTML = '';
  for (const msg of conversationHistory) {
    const div = document.createElement('div');
    div.className = msg.role === 'user' ? 'chat-msg chat-msg-user' : 'chat-msg chat-msg-assistant';

    const roleStrong = document.createElement('strong');
    roleStrong.textContent = msg.role === 'user' ? 'You: ' : 'Tutor: ';
    div.appendChild(roleStrong);

    const textSpan = document.createElement('span');
    textSpan.textContent = msg.content;
    div.appendChild(textSpan);

    box.appendChild(div);
  }
  box.scrollTop = box.scrollHeight;
}

function prefillTutorPrompt(promptText) {
  const input = document.getElementById('chat-input');
  if (!input) return;
  input.value = promptText;
  input.focus();
}

async function sendTutorQuestion() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('tutor-send-btn');
  const message = input.value.trim();
  if (!message) {
    setErrorBanner('tutor-error', 'Please enter a question for the tutor.');
    return;
  }

  const level = document.getElementById('level').value;
  const explanation_language = document.getElementById('explanation_language').value;
  const selectedQuestion = getSelectedQuestionPayload();
  const recentWindow = conversationHistory.slice(-MAX_RECENT_MESSAGES);

  setErrorBanner('tutor-error', '');
  setStatusBanner('tutor-status', 'Loading: Tutor is preparing a concise explanation...');
  if (sendBtn) {
    sendBtn.disabled = true;
    sendBtn.setAttribute('aria-busy', 'true');
  }
  input.value = '';

  try {
    const res = await fetch('/api/tutor', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify({
        level,
        explanation_language,
        question: selectedQuestion,
        message,
        history: recentWindow
      })
    });
    const data = await res.json();
    if (!res.ok) {
      setStatusBanner('tutor-status', '');
      setErrorBanner('tutor-error', data.detail || 'Failed to get tutor response.');
      input.value = message;
      return;
    }
    setStatusBanner('tutor-status', '');
    if (data.reply) {
      conversationHistory.push({ role: 'user', content: message });
      conversationHistory.push({ role: 'assistant', content: data.reply });
      if (conversationHistory.length > MAX_RECENT_MESSAGES) {
        conversationHistory = conversationHistory.slice(-MAX_RECENT_MESSAGES);
      }
      renderChat();
    }
  } catch (err) {
    setStatusBanner('tutor-status', '');
    setErrorBanner('tutor-error', 'Network error while contacting tutor.');
    input.value = message;
  } finally {
    if (sendBtn) {
      sendBtn.disabled = false;
      sendBtn.removeAttribute('aria-busy');
    }
  }
}

async function explainWord() {
  const wordInput = document.getElementById('word-input');
  const contextInput = document.getElementById('word-context-input');
  const levelSelect = document.getElementById('word-level');
  const langSelect = document.getElementById('word-explanation-language');
  const explainBtn = document.getElementById('word-explain-btn');

  const word = wordInput ? wordInput.value.trim() : '';
  const contextRaw = contextInput ? contextInput.value.trim() : '';
  const context = contextRaw ? contextRaw : null;
  const level = levelSelect ? levelSelect.value : 'N5';
  const explanation_language = langSelect ? langSelect.value : 'English';

  if (!word) {
    setErrorBanner('word-error', 'Please enter a Japanese word or expression to explain.');
    return;
  }

  setErrorBanner('word-error', '');
  setStatusBanner(
    'word-status',
    'Loading: Generating word meanings, original examples, verified past-exam check, and nuances...'
  );
  if (explainBtn) {
    explainBtn.disabled = true;
    explainBtn.setAttribute('aria-busy', 'true');
  }

  try {
    const res = await fetch('/api/words/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify({
        word,
        context,
        level,
        explanation_language
      })
    });
    const data = await res.json();
    if (!res.ok) {
      setStatusBanner('word-status', '');
      setErrorBanner('word-error', data.detail || 'Failed to explain word.');
      return;
    }
    currentWordExplanation = data;
    setStatusBanner('word-status', data.notice || 'Word explanation ready.');
    renderWordExplanation();
  } catch (err) {
    setStatusBanner('word-status', '');
    setErrorBanner('word-error', 'Network error while requesting word explanation.');
  } finally {
    if (explainBtn) {
      explainBtn.disabled = false;
      explainBtn.removeAttribute('aria-busy');
    }
  }
}

function renderWordExplanation() {
  const container = document.getElementById('word-explanation-results');
  if (!container) return;
  container.innerHTML = '';

  if (!currentWordExplanation) return;
  const data = currentWordExplanation;

  // 1. Meanings & Context Fit Section
  const meaningsSection = document.createElement('section');
  meaningsSection.id = 'word-meanings-section';
  meaningsSection.className = 'question-card word-subpanel';

  const meaningsHeader = document.createElement('h3');
  meaningsHeader.textContent = '4a. Meanings, Reading & Context Disambiguation';
  meaningsSection.appendChild(meaningsHeader);

  const headwordLine = document.createElement('p');
  const wordStrong = document.createElement('strong');
  wordStrong.setAttribute('lang', 'ja');
  wordStrong.className = 'jp-text';
  wordStrong.textContent = `${data.word} 【${data.reading}】 `;
  headwordLine.appendChild(wordStrong);

  const levelBadge = document.createElement('span');
  levelBadge.className = 'badge';
  levelBadge.textContent = `${data.level} — ${data.approximate_difficulty_label || 'Approximate JLPT difficulty'}`;
  headwordLine.appendChild(levelBadge);
  meaningsSection.appendChild(headwordLine);

  if (Array.isArray(data.alternate_readings) && data.alternate_readings.length > 0) {
    const altReadP = document.createElement('p');
    const altStrong = document.createElement('strong');
    altStrong.textContent = 'Alternate / Ambiguous Readings: ';
    altReadP.appendChild(altStrong);
    const altSpan = document.createElement('span');
    altSpan.setAttribute('lang', 'ja');
    altSpan.className = 'jp-text';
    altSpan.textContent = data.alternate_readings.join('、');
    altReadP.appendChild(altSpan);
    meaningsSection.appendChild(altReadP);
  }

  if (data.uncertainty_note || data.is_known_word === false) {
    const uncBox = document.createElement('div');
    uncBox.className = 'notice';
    uncBox.textContent =
      data.uncertainty_note ||
      'Note: This word or reading may be unrecognized, rare, or context-dependent.';
    meaningsSection.appendChild(uncBox);
  }

  if (data.context_fit_explanation) {
    const ctxBox = document.createElement('div');
    ctxBox.className = 'status-banner';
    const ctxStrong = document.createElement('strong');
    ctxStrong.textContent = 'Context Fit: ';
    ctxBox.appendChild(ctxStrong);
    ctxBox.appendChild(document.createTextNode(data.context_fit_explanation));
    meaningsSection.appendChild(ctxBox);
  }

  const meaningsList = document.createElement('ol');
  (data.meanings || []).forEach((m) => {
    const li = document.createElement('li');
    li.style.marginBottom = '0.65rem';

    const posBadge = document.createElement('span');
    posBadge.className = 'badge';
    posBadge.textContent = m.part_of_speech;
    li.appendChild(posBadge);

    const defSpan = document.createElement('span');
    defSpan.textContent = ` ${m.meaning}`;
    li.appendChild(defSpan);

    if (m.context_fit) {
      const fitP = document.createElement('div');
      fitP.style.fontSize = '0.88rem';
      fitP.textContent = `Context note: ${m.context_fit}`;
      li.appendChild(fitP);
    }

    if (Array.isArray(m.collocations) && m.collocations.length > 0) {
      const colP = document.createElement('div');
      colP.style.fontSize = '0.88rem';
      colP.textContent = `Common collocations: ${m.collocations.join(' / ')}`;
      li.appendChild(colP);
    }

    if (Array.isArray(m.grammar_patterns) && m.grammar_patterns.length > 0) {
      const gramP = document.createElement('div');
      gramP.style.fontSize = '0.88rem';
      gramP.textContent = `Grammar patterns: ${m.grammar_patterns.join(' / ')}`;
      li.appendChild(gramP);
    }

    meaningsList.appendChild(li);
  });
  meaningsSection.appendChild(meaningsList);
  container.appendChild(meaningsSection);

  // 2. Original AI-Generated Examples Section
  const origExamplesSection = document.createElement('section');
  origExamplesSection.id = 'word-original-examples-section';
  origExamplesSection.className = 'question-card word-subpanel';

  const origHeader = document.createElement('h3');
  origHeader.textContent = '4b. Original AI-Generated Examples (Unofficial)';
  origExamplesSection.appendChild(origHeader);

  const origLabelNote = document.createElement('p');
  origLabelNote.className = 'notice';
  origLabelNote.textContent =
    data.original_examples_label ||
    'Original AI-generated examples (unofficial study material, separate from verified past JLPT exams).';
  origExamplesSection.appendChild(origLabelNote);

  (data.meanings || []).forEach((m, mIdx) => {
    (m.examples || []).forEach((ex) => {
      const exBox = document.createElement('div');
      exBox.className = 'feedback-panel';
      exBox.style.marginTop = '0.5rem';

      const tagRow = document.createElement('div');
      const senseBadge = document.createElement('span');
      senseBadge.className = 'badge';
      senseBadge.textContent = `Meaning ${mIdx + 1} · ${ex.source_label || 'Original AI-generated example'}`;
      tagRow.appendChild(senseBadge);
      exBox.appendChild(tagRow);

      const jpLine = document.createElement('p');
      jpLine.setAttribute('lang', 'ja');
      jpLine.className = 'jp-text';
      jpLine.style.margin = '0.35rem 0 0.15rem';
      const jpStrong = document.createElement('strong');
      jpStrong.textContent = ex.japanese;
      jpLine.appendChild(jpStrong);
      exBox.appendChild(jpLine);

      const readLine = document.createElement('div');
      readLine.setAttribute('lang', 'ja');
      readLine.className = 'jp-text';
      readLine.style.fontSize = '0.88rem';
      readLine.textContent = `Reading (kana): ${ex.reading}`;
      exBox.appendChild(readLine);

      const transLine = document.createElement('div');
      transLine.style.fontSize = '0.92rem';
      transLine.textContent = `Translation: ${ex.translation}`;
      exBox.appendChild(transLine);

      const noteLine = document.createElement('div');
      noteLine.style.fontSize = '0.88rem';
      noteLine.textContent = `Usage note: ${ex.usage_note}`;
      exBox.appendChild(noteLine);

      origExamplesSection.appendChild(exBox);
    });
  });
  container.appendChild(origExamplesSection);

  // 3. Verified Examples from Past JLPT Exams Section (Curated Collection Only)
  const verifiedSection = document.createElement('section');
  verifiedSection.id = 'word-verified-examples-section';
  verifiedSection.className = 'question-card word-subpanel';

  const verHeader = document.createElement('h3');
  verHeader.textContent = '4c. Verified Examples from Past JLPT Exams (Curated Source Collection)';
  verifiedSection.appendChild(verHeader);

  const verStatus = document.createElement('p');
  verStatus.className = 'notice';
  verStatus.textContent =
    data.verified_examples_status ||
    'No verified past JLPT exam examples are available for this word in the curated source collection.';
  verifiedSection.appendChild(verStatus);

  const verifiedList = data.verified_past_exam_examples || [];
  verifiedList.forEach((vex) => {
    const vBox = document.createElement('div');
    vBox.className = 'passage-card';
    vBox.style.marginTop = '0.5rem';

    const vBadge = document.createElement('span');
    vBadge.className = 'badge';
    vBadge.textContent = 'Verified Past JLPT Exam Record (Curated Collection)';
    vBox.appendChild(vBadge);

    const vJp = document.createElement('p');
    vJp.setAttribute('lang', 'ja');
    vJp.className = 'jp-text';
    vJp.style.margin = '0.35rem 0 0.15rem';
    const vStrong = document.createElement('strong');
    vStrong.textContent = vex.japanese;
    vJp.appendChild(vStrong);
    vBox.appendChild(vJp);

    if (vex.reading) {
      const vRead = document.createElement('div');
      vRead.setAttribute('lang', 'ja');
      vRead.className = 'jp-text';
      vRead.style.fontSize = '0.88rem';
      vRead.textContent = `Reading: ${vex.reading}`;
      vBox.appendChild(vRead);
    }

    const provLine = document.createElement('div');
    provLine.style.fontSize = '0.86rem';
    const metaParts = [
      `Source: ${vex.source_title}`,
      `Ref: ${vex.source_reference}`
    ];
    if (vex.exam_year) metaParts.push(`Year: ${vex.exam_year}`);
    if (vex.exam_level) metaParts.push(`Level: ${vex.exam_level}`);
    if (vex.question_or_page_ref) metaParts.push(`Item: ${vex.question_or_page_ref}`);
    provLine.textContent = metaParts.join(' · ');
    vBox.appendChild(provLine);

    if (vex.source_url && /^https?:\/\//i.test(vex.source_url)) {
      const linkLine = document.createElement('div');
      linkLine.style.fontSize = '0.84rem';
      const link = document.createElement('a');
      link.href = vex.source_url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = vex.source_url;
      linkLine.appendChild(link);
      vBox.appendChild(linkLine);
    }

    verifiedSection.appendChild(vBox);
  });
  container.appendChild(verifiedSection);

  // 4. Nuances, Register & Usage Contrasts Section
  const nuancesSection = document.createElement('section');
  nuancesSection.id = 'word-nuances-section';
  nuancesSection.className = 'question-card word-subpanel';

  const nuancesHeader = document.createElement('h3');
  nuancesHeader.textContent = '4d. Nuances, Register, Synonyms & Usage Contrasts';
  nuancesSection.appendChild(nuancesHeader);

  if (data.register_and_formality) {
    const regP = document.createElement('p');
    const regStrong = document.createElement('strong');
    regStrong.textContent = 'Formality, Register & Typical Situations: ';
    regP.appendChild(regStrong);
    regP.appendChild(document.createTextNode(data.register_and_formality));
    nuancesSection.appendChild(regP);
  }

  const nuanceUl = document.createElement('ul');
  (data.nuances || []).forEach((nText) => {
    const li = document.createElement('li');
    li.textContent = nText;
    nuanceUl.appendChild(li);
  });
  nuancesSection.appendChild(nuanceUl);

  if (Array.isArray(data.synonyms_and_contrasts) && data.synonyms_and_contrasts.length > 0) {
    const synTitle = document.createElement('p');
    const synStrong = document.createElement('strong');
    synStrong.textContent = 'Synonyms & Easily Confused Words:';
    synTitle.appendChild(synStrong);
    nuancesSection.appendChild(synTitle);

    const synUl = document.createElement('ul');
    data.synonyms_and_contrasts.forEach((syn) => {
      const li = document.createElement('li');
      li.textContent = `${syn.term}: ${syn.comparison}`;
      synUl.appendChild(li);
    });
    nuancesSection.appendChild(synUl);
  }

  if (Array.isArray(data.natural_vs_unnatural) && data.natural_vs_unnatural.length > 0) {
    const natTitle = document.createElement('p');
    const natStrong = document.createElement('strong');
    natStrong.textContent = 'Natural vs. Unnatural Usage:';
    natTitle.appendChild(natStrong);
    nuancesSection.appendChild(natTitle);

    const natUl = document.createElement('ul');
    data.natural_vs_unnatural.forEach((pair) => {
      const li = document.createElement('li');
      li.textContent = `✓ Natural: ${pair.natural_example} | ✗ Unnatural: ${pair.unnatural_example} — ${pair.explanation}`;
      natUl.appendChild(li);
    });
    nuancesSection.appendChild(natUl);
  }

  container.appendChild(nuancesSection);

  // 5. AI-generated follow-up questions (generated in the same call as the explanation)
  if (Array.isArray(data.follow_up_questions) && data.follow_up_questions.length > 0) {
    const followUpSection = document.createElement('section');
    followUpSection.id = 'word-follow-up-section';
    followUpSection.className = 'question-card word-subpanel';

    const followUpHeader = document.createElement('h3');
    followUpHeader.textContent = '4e. Suggested Follow-Up Questions (AI-Generated)';
    followUpSection.appendChild(followUpHeader);

    const followUpList = document.createElement('ul');
    data.follow_up_questions.forEach((questionText) => {
      const li = document.createElement('li');
      li.textContent = questionText;
      followUpList.appendChild(li);
    });
    followUpSection.appendChild(followUpList);
    container.appendChild(followUpSection);
  }
}

function clearWordExplanation() {
  currentWordExplanation = null;
  setStatusBanner('word-status', '');
  setErrorBanner('word-error', '');
  const wordInput = document.getElementById('word-input');
  const contextInput = document.getElementById('word-context-input');
  const resultsEl = document.getElementById('word-explanation-results');
  if (wordInput) wordInput.value = '';
  if (contextInput) contextInput.value = '';
  if (resultsEl) resultsEl.innerHTML = '';
}

function resetSession() {
  currentPracticeSet = null;
  selectedAnswers = {};
  isQuizSubmitted = false;
  activeTutorQuestionIndex = 0;
  conversationHistory = [];
  currentWordExplanation = null;
  sessionSummary = {
    setsCompleted: 0,
    answeredCount: 0,
    correctCount: 0,
    missedTopics: []
  };
  setStatusBanner('quiz-status', '');
  setErrorBanner('quiz-error', '');
  setStatusBanner('tutor-status', '');
  setErrorBanner('tutor-error', '');
  setStatusBanner('word-status', '');
  setErrorBanner('word-error', '');
  document.getElementById('quiz-container').innerHTML = '';
  document.getElementById('quiz-score-summary').innerHTML = '';
  document.getElementById('quiz-actions').style.display = 'none';
  document.getElementById('chat-input').value = '';
  clearWordExplanation();
  updateTutorQuestionSelector();
  renderChat();
  renderSessionSummary();
}
