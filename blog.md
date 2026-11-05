---
layout: default
title: "Blog"
permalink: /blog/
---

# Blog

{% if site.posts.size > 0 %}
<div class="entries">
  {% for post in site.posts %}
  <div class="entry">
    <p class="entry-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></p>
    <p class="entry-meta">{{ post.date | date: "%B %-d, %Y" }}{% if post.description %} · {{ post.description }}{% endif %}</p>
  </div>
  {% endfor %}
</div>
{% else %}
<p class="note">No posts yet.</p>
{% endif %}
