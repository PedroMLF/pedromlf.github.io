---
layout: default
title: ""
description: ""
---

<div class="intro">
  <div class="intro-text">
    <h1>{{ site.title }}</h1>
    <p class="lede">{{ site.tagline }}</p>

    <p>I am a final-year PhD candidate at the University of
      Amsterdam (UvA), supervised by prof. <a href="https://wilkeraziz.github.io/">Wilker Aziz</a>
      and prof. <a href="https://ivan-titov.org/">Ivan Titov</a>, as part of the Mercury Machine
      Learning Lab with Booking.com. My main research interests are related to the safety and
      trustworthiness of language models, in particular the faithfulness and monitorability of their reasoning. More broadly,
      I am interested in interpretability, post-training methods, and evaluation.</p>

    <p>In 2025 I was a research intern at NAVER Labs Europe, working on fine-tuning LLMs to
      satisfy rare constraints, a project that received an internal Distinguished Award.
      Before my PhD, I worked in industry building ML systems at Priberam, Unbabel and Farfetch.
      I obtained an integrated master's degree in Electrical and Computer
      Engineering from IST, University of Lisbon, with a thesis at Unbabel supervised by
      prof. <a href="https://andre-martins.github.io/">André Martins</a>.</p>

    <p>I am on the job market for ML research and engineering roles, starting in early 2027.</p>

    <p class="intro-links">
      {% if site.email_reversed %}<a href="#" class="email-link" data-e="{{ site.email_reversed }}">Email</a> · {% endif %}{% for link in site.links %}<a href="{{ link.url }}">{{ link.name }}</a>{% unless forloop.last %} · {% endunless %}{% endfor %}
    </p>
  </div>

  <img class="photo" src="{{ '/assets/img/photo.jpg' | relative_url }}" alt="Photo of {{ site.title }}">
</div>

## Experience

<ul class="timeline">
  <li><span class="tl-date">2022 – 2026</span>
    <div class="tl-body"><p>PhD Candidate, University of Amsterdam <span class="tl-aside">(industry-funded by Booking.com)</span></p></div></li>
  <li><span class="tl-date">2025</span>
    <div class="tl-body"><p>Research Intern, NAVER Labs Europe</p></div></li>
  <li><span class="tl-date">2020 – 2021</span>
    <div class="tl-body"><p>Research Scientist, Priberam</p></div></li>
  <li><span class="tl-date">2019 – 2020</span>
    <div class="tl-body"><p>Junior AI Researcher, Unbabel</p></div></li>
  <li><span class="tl-date">2018 – 2019</span>
    <div class="tl-body"><p>Junior Data Scientist, Farfetch</p></div></li>
  <li><span class="tl-date">2017 – 2018</span>
    <div class="tl-body"><p>Undergraduate Research Intern, Unbabel</p></div></li>
</ul>

## Publications

<div class="entries">

  <div class="entry">
    <p class="entry-title"><a href="https://aclanthology.org/2026.blackboxnlp-1.30/">Does Out-of-Sight Equal Out-of-Mind in CoT Monitorability?</a></p>
    <p><strong>Pedro Ferreira</strong>, Wilker Aziz, Ivan Titov.</p>
    <p class="entry-meta"><em>BlackboxNLP Workshop at EMNLP 2026</em>, Hungary 🇭🇺</p>
    <p class="entry-note">🏆 Best Paper (main track). Oral presentation.</p>
  </div>

  <div class="entry">
    <p class="entry-title"><a href="https://iclr.cc/virtual/2026/poster/10007460">Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations</a></p>
    <p><strong>Pedro Ferreira</strong>, Wilker Aziz, Ivan Titov.</p>
    <p class="entry-meta"><em>ICLR 2026</em>, Rio de Janeiro, Brazil 🇧🇷</p>
    <p class="entry-note">🏆 Outstanding Paper at the <a href="https://actionable-interpretability.github.io/2025/">Workshop on Actionable Interpretability</a>, ICML 2025.</p>
  </div>

  <div class="entry">
    <p class="entry-title"><a href="https://aclanthology.org/2025.coling-main.436/">Explanation Regularisation through the Lens of Attributions</a></p>
    <p><strong>Pedro Ferreira</strong>, Ivan Titov, Wilker Aziz.</p>
    <p class="entry-meta"><em>COLING 2025</em>, Abu Dhabi, UAE 🇦🇪</p>
    <p class="entry-note">Oral presentation.</p>
  </div>

  <div class="entry">
    <p class="entry-title"><a href="https://aclanthology.org/2021.bsnlp-1.10/">Priberam Labs at the 3rd Shared Task on SlavNER</a></p>
    <p><strong>Pedro Ferreira</strong>, Rúben Cardoso, Afonso Mendes.</p>
    <p class="entry-meta"><em>BSNLP Workshop at EACL 2021</em>, Kyiv, Ukraine 🇺🇦 (virtual)</p>
    <p class="entry-note">🏆 1st place, SlavNER Shared Task.</p>
  </div>

  <div class="entry">
    <p class="entry-title"><a href="https://aclanthology.org/P18-2059/">Sparse and Constrained Attention for Neural Machine Translation</a></p>
    <p>Chaitanya Malaviya, <strong>Pedro Ferreira</strong>, André F. T. Martins.</p>
    <p class="entry-meta"><em>ACL 2018</em>, Melbourne, Australia 🇦🇺</p>
  </div>

</div>

## Talks

<ul class="timeline">
  <li><span class="tl-date">2026</span>
    <div class="tl-body"><p>Efficient Reasoning in Large Language Models</p>
      <p class="tl-note">Internal talk, Booking.com.</p></div></li>
  <li><span class="tl-date">2025</span>
    <div class="tl-body"><p>Reward Hacking in Large Language Models</p>
      <p class="tl-note">Internal talk, Booking.com.</p></div></li>
  <li><span class="tl-date">2023</span>
    <div class="tl-body"><p>Slot Attention as a Tool for Interpretability and Better Representation Disentanglement</p>
      <p class="tl-note">Booking.com Science and Analytics Conference.</p></div></li>
</ul>

## Teaching & Supervision {#teaching}

<ul class="timeline">
  <li><span class="tl-date">2026</span>
    <div class="tl-body"><p>Co-supervisor, MInf thesis, University of Edinburgh</p>
      <p class="tl-note">Lyle Buist, <em>Using Linear Probes to Detect and Reduce Unfaithful CoT-Reasoning in LLM Responses</em>.</p></div></li>
  <li><span class="tl-date">2024</span>
    <div class="tl-body"><p>Invited Lecturer, University of Amsterdam</p>
      <ul class="tl-sub">
        <li>Language, Speech, and Dialogue Processing (BSc AI)</li>
      </ul></div></li>
  <li><span class="tl-date">2022 – 2024</span>
    <div class="tl-body"><p>Teaching Assistant, University of Amsterdam</p>
      <ul class="tl-sub">
        <li>Natural Language Processing I (MSc AI, 2023; 2024)</li>
        <li>Language, Speech, and Dialogue Processing (BSc AI, 2023)</li>
        <li>Basic Probability Theory (MSc Logic, 2022)</li>
      </ul></div></li>
  <li><span class="tl-date">2019 – 2023</span>
    <div class="tl-body"><p>Instructor, Lisbon Machine Learning School (LxMLS)</p>
      <ul class="tl-sub">
        <li>Labs team, 2019, 2021 and 2023 editions</li>
      </ul></div></li>
  <li><span class="tl-date">2018</span>
    <div class="tl-body"><p>Instructor, Lisbon Data Science Academy</p>
      <ul class="tl-sub">
        <li>Instructor of the 2018 edition, with ~50 students</li>
        <li>Organized the NLP part of the curriculum, together with three other instructors</li>
      </ul></div></li>
</ul>
