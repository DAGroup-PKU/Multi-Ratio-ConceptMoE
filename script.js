"use strict";

const compressionModes = {
  cr2: { ratios: [2, 2, 2], note: "Fixed CR2: Turn 1 → CR2, Turn 2 → CR2, Turn 3 → CR2. Every turn uses the same ratio." },
  cr4: { ratios: [4, 4, 4], note: "Fixed CR4: Turn 1 → CR4, Turn 2 → CR4, Turn 3 → CR4. Every turn uses the same ratio." },
  cr8: { ratios: [8, 8, 8], note: "Fixed CR8: Turn 1 → CR8, Turn 2 → CR8, Turn 3 → CR8. Every turn uses the same ratio." },
  mixed: { ratios: [2, 8, 4], note: "Mixed: Turn 1 → CR2, Turn 2 → CR8, Turn 3 → CR4. Different turns use different ratios within one conversation." }
};

function setCompressionMode(mode) {
  const { ratios, note } = compressionModes[mode];
  document.querySelectorAll("[data-mode]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
  document.querySelectorAll("[data-turn]").forEach((turn, index) => {
    const ratio = ratios[index];
    const count = 24 / ratio;
    const label = turn.querySelector(".ratio-label");
    label.textContent = `CR${ratio}`;
    label.className = `ratio-label cr${ratio}`;
    const tokens = turn.querySelector(".token-row");
    const concepts = turn.querySelector(".concept-row");
    tokens.replaceChildren(...Array.from({ length: 24 }, () => document.createElement("i")));
    concepts.replaceChildren(...Array.from({ length: count }, () => document.createElement("i")));
    concepts.className = `concept-row cr${ratio}`;
    turn.querySelector(".concept-caption").textContent = `24 ÷ ${ratio} = ${count} concepts`;
  });
  document.getElementById("mode-note").textContent = note;
  const counts = ratios.map(ratio => 24 / ratio);
  document.getElementById("history-stream").replaceChildren(...ratios.map((ratio, index) => {
    const group = document.createElement("div");
    group.className = `history-group cr${ratio}`;
    group.style.gridColumn = `span ${counts[index]}`;
    group.setAttribute("role", "img");
    group.setAttribute("aria-label", `Turn ${index + 1}: ${counts[index]} concepts at CR${ratio}`);
    const label = document.createElement("span");
    label.textContent = `T${index + 1}`;
    label.className = "history-label";
    const blocks = document.createElement("div");
    blocks.className = `concept-row cr${ratio}`;
    blocks.style.gridTemplateColumns = `repeat(${counts[index]}, minmax(0, 1fr))`;
    blocks.setAttribute("aria-hidden", "true");
    blocks.replaceChildren(...Array.from({ length: counts[index] }, () => document.createElement("i")));
    group.append(label, blocks);
    return group;
  }));
  document.getElementById("history-count").textContent = `After three turns: ${counts.join(" + ")} = ${counts.reduce((a, b) => a + b, 0)} concepts in one sequence.`;
}
document.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => setCompressionMode(button.dataset.mode)));
setCompressionMode("mixed");

// Reported results, transcribed from the manuscript's experimental tables.
const datasets = {
  codechat: {
    label: "CodeChat", metric: "Combined Joint accuracy", accuracy: "43.78", gain: "+5.87 pp", compute: "1.0999", saving: "41.1% fewer FLOPs",
    rows: [
      ["MoE Large", "37.91", "1.8662"], ["MoE Medium", "27.09", "1.1198"], ["MoE Small", "18.68", "0.7334"],
      ["Multi-Ratio CR2", "43.28", "1.3072"], ["Multi-Ratio CR4", "35.37", "0.9529"], ["Multi-Ratio CR8", "29.29", "0.7795"],
      ["Multi-Ratio Mixed", "43.78", "1.0999"],
      ["SingleCR2 specialist", "33.89", "2.2290"], ["SingleCR4 specialist", "24.60", "1.2784"], ["SingleCR8 specialist", "18.02", "0.7326"]
    ]
  },
  cosql: {
    label: "CoSQL", metric: "SQLite execution accuracy", accuracy: "51.84", gain: "+2.68 pp", compute: "0.3319", saving: "12.3% fewer FLOPs",
    rows: [
      ["MoE Large", "49.16", "0.3785"], ["MoE Medium", "25.52", "0.2283"], ["MoE Small", "14.00", "0.1502"],
      ["Multi-Ratio CR2", "54.72", "0.3788"], ["Multi-Ratio CR4", "48.26", "0.2418"], ["Multi-Ratio CR8", "41.11", "0.1863"],
      ["Multi-Ratio Mixed", "51.84", "0.3319"],
      ["SingleCR2 specialist", "54.02", "0.4027"], ["SingleCR4 specialist", "42.50", "0.2873"], ["SingleCR8 specialist", "33.37", "0.2427"]
    ]
  }
};

function setDataset(key) {
  const data = datasets[key];
  document.querySelectorAll("[data-dataset]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.dataset === key)));
  document.getElementById("accuracy-metric").textContent = data.metric;
  const accuracyUnit = document.createElement("span"); accuracyUnit.textContent = "%";
  document.getElementById("mixed-accuracy").replaceChildren(data.accuracy, accuracyUnit);
  const computeUnit = document.createElement("span"); computeUnit.textContent = " PFLOPs";
  document.getElementById("mixed-compute").replaceChildren(data.compute, computeUnit);
  document.getElementById("accuracy-gain").textContent = data.gain;
  document.getElementById("compute-saving").textContent = data.saving;
  document.getElementById("dataset-caption").textContent = `${data.label} · terminal-turn workload`;
  document.getElementById("comparison-body").replaceChildren(...data.rows.map((row, index) => {
    const tr = document.createElement("tr");
    if (index === 3 || index === 7) tr.classList.add("group-start");
    if (index === 6) tr.classList.add("highlight-row");
    row.forEach((value, column) => {
      const cell = document.createElement(column === 0 ? "th" : "td");
      if (column === 0) cell.scope = "row";
      cell.textContent = value;
      if (index === 6 && column === 0) { const tag = document.createElement("span"); tag.textContent = "Ours"; cell.append(" ", tag); }
      tr.append(cell);
    });
    return tr;
  }));
}
document.querySelectorAll("[data-dataset]").forEach(button => button.addEventListener("click", () => setDataset(button.dataset.dataset)));

const figureDialog = document.getElementById("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach(button => button.addEventListener("click", () => {
  const source = button.querySelector("img");
  const expanded = document.getElementById("expanded-figure");
  expanded.src = button.dataset.zoom;
  expanded.alt = source.alt;
  document.getElementById("expanded-caption").textContent = button.closest("figure").querySelector("figcaption").textContent;
  figureDialog.showModal();
  document.body.classList.add("figure-open");
}));
document.getElementById("close-figure").addEventListener("click", () => figureDialog.close());
figureDialog.addEventListener("click", event => { if (event.target === figureDialog) figureDialog.close(); });
figureDialog.addEventListener("close", () => document.body.classList.remove("figure-open"));

const copyButton = document.getElementById("copy-citation");
let copyResetTimer;
copyButton.addEventListener("click", async () => {
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(document.getElementById("bibtex").textContent);
    copyButton.querySelector("span").textContent = "Copied!";
    status.textContent = "BibTeX copied to clipboard.";
    clearTimeout(copyResetTimer);
    copyResetTimer = setTimeout(() => { copyButton.querySelector("span").textContent = "Copy BibTeX"; status.textContent = ""; }, 2500);
  } catch {
    status.textContent = "Copy is unavailable here. Select the BibTeX text to copy it manually.";
  }
});

const scrollButton = document.getElementById("scrollBtn");
function updateScrollButton() {
  scrollButton.classList.toggle("visible", window.scrollY > 300);
}
window.addEventListener("scroll", updateScrollButton, { passive: true });
updateScrollButton();
