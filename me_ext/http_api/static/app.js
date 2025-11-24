async function loadJson(url) {
  const resp = await fetch(url);
  return resp.json();
}

async function loadStatus() {
  try {
    const data = await loadJson("/status");
    document.getElementById("status-text").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    document.getElementById("status-text").textContent = "加载失败";
  }
}

async function loadExperiments() {
  try {
    const data = await loadJson("/experiments/recent");
    document.getElementById("experiments-text").textContent = JSON.stringify(data, null, 2);
  } catch {
    document.getElementById("experiments-text").textContent = "加载失败";
  }
}

async function loadNotebook() {
  try {
    const data = await loadJson("/notebook/recent");
    document.getElementById("notebook-text").textContent = JSON.stringify(data, null, 2);
  } catch {
    document.getElementById("notebook-text").textContent = "加载失败";
  }
}

async function loadComparison() {
  try {
    const data = await loadJson("/report/comparison");
    document.getElementById("comparison-text").textContent = JSON.stringify(data, null, 2);
  } catch {
    document.getElementById("comparison-text").textContent = "加载失败";
  }
}

async function loadFigures() {
  document.getElementById("brain-graph-img").src = "/plots/brain_graph.png";
  document.getElementById("experiment-curve-img").src = "/plots/experiment_curve.png";
}

window.addEventListener("load", () => {
  loadStatus();
  loadExperiments();
  loadNotebook();
  loadComparison();
  loadFigures();
});
