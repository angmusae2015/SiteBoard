"use strict";

const form = document.querySelector("#ngram-form");
const query = document.querySelector("#ngram-query");
const size = document.querySelector("#ngram-size");
const errorMessage = document.querySelector("#ngram-error");
const status = document.querySelector("#ngram-status");
const results = document.querySelector("#ngram-results");
const pageSize = 50;
let pending;

const columns = Array.from(results.querySelectorAll("[data-method]"), (section) => {
  const column = {
    method: section.dataset.method,
    list: section.querySelector("ol"),
    count: section.querySelector(".count-pill"),
    more: section.querySelector(".ngram-more"),
    rows: [],
    shown: 0,
  };
  column.more.addEventListener("click", () => appendPage(column));
  return column;
});

function appendPage(column) {
  const next = column.rows.slice(column.shown, column.shown + pageSize);
  for (const row of next) {
    const item = document.createElement("li");
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const score = document.createElement("strong");
    const original = document.createElement("p");
    const meta = document.createElement("small");
    score.textContent = `${(row.score * 100).toFixed(2)}%`;
    original.textContent = row.x;
    meta.textContent = `${row.site} / 사진 ${row.image_id}`;
    summary.append(score, original, meta);

    const answer = document.createElement("div");
    answer.className = "ngram-answer";
    const heading = document.createElement("div");
    heading.className = "ngram-answer-heading";
    const label = document.createElement("b");
    label.textContent = "정답";
    const copyStatus = document.createElement("span");
    copyStatus.className = "ngram-copy-status";
    copyStatus.setAttribute("role", "status");
    heading.append(
      label,
      copyButton("JSON", JSON.stringify(row.y), copyStatus),
      copyButton("원문", row.x, copyStatus),
      copyStatus,
    );
    const works = document.createElement("div");
    works.className = "work-items";
    for (const work of row.y) {
      const entry = document.createElement("span");
      entry.className = "work-item";
      entry.textContent = [work.workType, work.spec].filter(Boolean).join(" ");
      const quantity = document.createElement("b");
      quantity.textContent = `× ${work.quantity}`;
      entry.append(quantity);
      works.append(entry);
    }
    answer.append(heading, works);
    details.append(summary, answer);
    item.append(details);
    column.list.append(item);
  }
  column.shown += next.length;
  column.more.hidden = column.shown >= column.rows.length;
}

function copyButton(label, text, feedback) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ngram-copy";
  button.textContent = `${label} 복사`;
  button.addEventListener("click", async () => {
    feedback.textContent = "";
    try {
      await navigator.clipboard.writeText(text);
      feedback.textContent = `${label} 복사됨`;
    } catch {
      feedback.textContent = `${label} 복사 실패`;
    }
  });
  return button;
}

function clearColumns() {
  for (const column of columns) {
    column.list.replaceChildren();
    column.count.hidden = true;
    column.more.hidden = true;
    column.shown = 0;
    column.rows = [];
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  pending?.abort();
  const request = new AbortController();
  pending = request;
  clearColumns();
  errorMessage.hidden = true;
  query.removeAttribute("aria-invalid");
  results.setAttribute("aria-busy", "true");
  status.textContent = "검색 중";

  try {
    const params = new URLSearchParams({ q: query.value, n: size.value });
    const response = await fetch(`/api/ngram?${params}`, { signal: request.signal });
    const data = await response.json();
    if (pending !== request) return;
    if (!response.ok) {
      if (response.status === 400) query.setAttribute("aria-invalid", "true");
      throw new Error(data.error || "검색에 실패했습니다.");
    }
    for (const column of columns) {
      column.rows = data[column.method];
      column.count.textContent = `${column.rows.length.toLocaleString("ko-KR")}건`;
      column.count.hidden = false;
      if (column.rows.length) {
        appendPage(column);
      } else {
        const empty = document.createElement("li");
        empty.className = "empty";
        empty.textContent = "검색 결과 없음";
        column.list.append(empty);
      }
    }
  } catch (error) {
    if (pending !== request) return;
    errorMessage.textContent = error.message;
    errorMessage.hidden = false;
  } finally {
    if (pending === request) {
      results.removeAttribute("aria-busy");
      status.textContent = "";
    }
  }
});
