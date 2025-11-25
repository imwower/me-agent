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

async function runTask() {
  try {
    const resp = await fetch("/task/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenario: "real_task",
        input: {
          text: "请根据结构化指标判断是否需要报警",
          image_path: "tests/data/dummy.png",
          structured: { cpu: 0.8, mem: 0.9 },
        },
      }),
    });
    const data = await resp.json();
    document.getElementById("task-run-text").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    document.getElementById("task-run-text").textContent = "调用失败";
  }
}

async function runTrain() {
  try {
    const resp = await fetch("/train/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "snn", max_steps: 10 }),
    });
    const data = await resp.json();
    document.getElementById("train-run-text").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    document.getElementById("train-run-text").textContent = "调用失败";
  }
}

window.addEventListener("load", () => {
  loadStatus();
  loadExperiments();
  loadNotebook();
  loadComparison();
  loadFigures();
  document.getElementById("run-task-btn").addEventListener("click", runTask);
  document.getElementById("run-train-btn").addEventListener("click", runTrain);
});
