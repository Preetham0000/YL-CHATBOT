import React from "https://esm.sh/react@18.3.1";
import { renderToStaticMarkup } from "https://esm.sh/react-dom@18.3.1/server";
import ReactMarkdown from "https://esm.sh/react-markdown@9?bundle";
import remarkGfm from "https://esm.sh/remark-gfm@4";

const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const log = document.getElementById("chat-log");
const template = document.getElementById("message-template");
const imageGrid = document.getElementById("image-grid");

const appendMessage = (author, text, { markdown = false } = {}) => {
  const clone = template.content.cloneNode(true);
  clone.querySelector(".meta").textContent = author;
  const content = clone.querySelector(".content");
  if (markdown) {
    const html = renderToStaticMarkup(
      React.createElement(ReactMarkdown, { remarkPlugins: [remarkGfm] }, text)
    );
    content.innerHTML = html;
  } else {
    content.textContent = text;
  }
  log.appendChild(clone);
  log.scrollTop = log.scrollHeight;
};

const renderImages = (images) => {
  imageGrid.innerHTML = "";
  images.forEach((url, idx) => {
    const figure = document.createElement("figure");
    const img = document.createElement("img");
    img.src = url;
    img.alt = `Referenced asset ${idx + 1}`;
    figure.appendChild(img);
    imageGrid.appendChild(figure);
  });
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) {
    return;
  }
  appendMessage("You", question);
  input.value = "";
  appendMessage("Bot", "Thinking...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      throw new Error("Request failed");
    }
    const data = await response.json();
    log.lastElementChild.remove();
    appendMessage("Bot", data.answer, { markdown: true });
    renderImages(data.images || []);
  } catch (error) {
    log.lastElementChild.remove();
    appendMessage("Bot", "Something went wrong. Please try again.");
    console.error(error);
  }
});
