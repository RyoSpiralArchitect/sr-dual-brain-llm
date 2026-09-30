/* A read-only, dependency-free view of one saved dialogue flow. */
(function () {
  "use strict";

  const MAX_STEPS = 80;
  const MAX_STAGES = 10;
  const MAX_CONTENT = 2400;
  const MAX_META_FIELDS = 12;
  const MAX_META_VALUE = 240;
  const PLAY_INTERVAL_MS = 1400;

  function element(tag, className, label) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (label != null) node.textContent = label;
    return node;
  }

  function limited(value, length) {
    const text = value == null ? "" : String(value);
    return text.length > length
      ? `${text.slice(0, length)}… (${text.length - length} more characters)`
      : text;
  }

  function roleKind(role) {
    const name = role.toLowerCase();
    if (name.includes("left")) return "left";
    if (name.includes("right") || name.includes("critic")) return "right";
    if (name.includes("callosum") || name.includes("coordinator")) return "bridge";
    if (name.includes("integrator") || name.includes("cerebellum")) return "integrator";
    return "other";
  }

  function shallowValue(value) {
    if (value == null) return String(value);
    if (Array.isArray(value)) return `[array of ${value.length}]`;
    if (typeof value === "object") return "[object]";
    return limited(value, MAX_META_VALUE);
  }

  function create(root) {
    if (!root || typeof root.replaceChildren !== "function") {
      throw new TypeError("TraceTheater.create requires a DOM element");
    }

    let qid = null;
    let steps = [];
    let totalSteps = 0;
    let stepButtons = [];
    let selected = 0;
    let timer = null;
    let destroyed = false;
    const motion = typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-reduced-motion: reduce)")
      : null;

    const shell = element("section", "tt");
    shell.setAttribute("aria-label", "Trace theater");
    const header = element("div", "tt__header");
    const heading = element("div", "tt__heading");
    heading.appendChild(element("span", "tt__eyebrow", "TRACE THEATER"));
    heading.appendChild(element("strong", "tt__title", "A turn, frame by frame"));
    const count = element("span", "tt__count");
    const announcement = element("span", "tt__sr-only");
    announcement.setAttribute("aria-live", "polite");
    announcement.setAttribute("aria-atomic", "true");
    header.appendChild(heading);
    header.appendChild(count);
    header.appendChild(announcement);

    const controls = element("div", "tt__controls");
    const previous = element("button", "tt__control", "← Previous");
    const play = element("button", "tt__control tt__control--play", "▶ Play");
    const next = element("button", "tt__control", "Next →");
    for (const button of [previous, play, next]) button.type = "button";
    play.setAttribute("aria-pressed", "false");
    controls.appendChild(previous);
    controls.appendChild(play);
    controls.appendChild(next);

    const progress = element("div", "tt__progress");
    const progressFill = element("div", "tt__progress-fill");
    progress.appendChild(progressFill);
    const rail = element("ol", "tt__rail");
    rail.setAttribute("aria-label", "Dialogue steps");
    const frame = element("div", "tt__frame");
    const empty = element("p", "tt__empty", "No dialogue steps in this trace yet.");
    const frameTop = element("div", "tt__frame-top");
    const role = element("span", "tt__role");
    const phase = element("strong", "tt__phase");
    frameTop.appendChild(role);
    frameTop.appendChild(phase);
    const details = element("details", "tt__details");
    details.appendChild(element("summary", "", "View step content and metadata"));
    const detailBody = element("div", "tt__detail-body");
    details.appendChild(detailBody);
    frame.appendChild(empty);
    frame.appendChild(frameTop);
    frame.appendChild(details);

    const architecture = element("div", "tt__architecture");
    architecture.appendChild(element("span", "tt__architecture-label", "Architecture path"));
    const stages = element("div", "tt__stages");
    architecture.appendChild(stages);

    shell.appendChild(header);
    shell.appendChild(controls);
    shell.appendChild(progress);
    shell.appendChild(rail);
    shell.appendChild(frame);
    shell.appendChild(architecture);
    root.replaceChildren(shell);

    function reducedMotion() {
      return Boolean(motion && motion.matches);
    }

    function stopPlayback() {
      if (timer != null) clearInterval(timer);
      timer = null;
      play.textContent = "▶ Play";
      play.setAttribute("aria-pressed", "false");
    }

    function showStep() {
      const hasSteps = steps.length > 0;
      const entry = hasSteps ? steps[selected] : null;
      const step = entry?.step;
      const stepNumber = entry ? entry.index + 1 : 0;
      const omitted = totalSteps - steps.length;
      count.textContent = hasSteps
        ? `Step ${stepNumber} of ${totalSteps}${omitted ? ` · ${omitted} omitted` : ""}`
        : "No steps";
      announcement.textContent = hasSteps
        ? `Step ${stepNumber} of ${totalSteps}: ${limited(step.role || "unknown", 32)}, ${limited(step.phase || "step", 80).replaceAll("_", " ")}${omitted ? `. ${omitted} middle steps omitted` : ""}`
        : empty.textContent;
      progressFill.style.width = hasSteps ? `${(stepNumber / totalSteps) * 100}%` : "0%";
      previous.disabled = !hasSteps || selected === 0;
      next.disabled = !hasSteps || selected === steps.length - 1;
      play.disabled = steps.length < 2 || reducedMotion();
      play.title = reducedMotion() ? "Playback is paused for reduced motion"
        : omitted ? `Play visible steps (${omitted} middle steps omitted)` : "Play dialogue steps";
      empty.hidden = hasSteps;
      frameTop.hidden = !hasSteps;
      details.hidden = !hasSteps;

      for (let i = 0; i < stepButtons.length; i++) {
        const button = stepButtons[i];
        const active = i === selected;
        button.className = `tt__step${active ? " tt__step--active" : ""}`;
        if (active) button.setAttribute("aria-current", "step");
        else button.removeAttribute("aria-current");
      }
      if (!step) {
        detailBody.replaceChildren();
        return;
      }

      const stepRole = limited(step.role || "unknown", 32);
      role.textContent = stepRole;
      role.setAttribute("data-kind", roleKind(stepRole));
      phase.textContent = limited(step.phase || "step", 80).replaceAll("_", " ");
      details.open = false;
      detailBody.replaceChildren();

      if (step.content != null && String(step.content).trim()) {
        const contentBlock = element("div", "tt__detail-section");
        contentBlock.appendChild(element("span", "tt__detail-label", "Content"));
        contentBlock.appendChild(element("pre", "tt__content", limited(step.content, MAX_CONTENT)));
        detailBody.appendChild(contentBlock);
      }
      if (step.meta && typeof step.meta === "object" && !Array.isArray(step.meta)) {
        const keys = Object.keys(step.meta).slice(0, MAX_META_FIELDS);
        if (keys.length) {
          const metaBlock = element("div", "tt__detail-section");
          metaBlock.appendChild(element("span", "tt__detail-label", "Metadata"));
          const list = element("dl", "tt__meta");
          for (const key of keys) {
            list.appendChild(element("dt", "", limited(key, 80)));
            list.appendChild(element("dd", "", shallowValue(step.meta[key])));
          }
          if (Object.keys(step.meta).length > keys.length) {
            list.appendChild(element("dt", "", "More fields"));
            list.appendChild(element("dd", "", `${Object.keys(step.meta).length - keys.length} hidden`));
          }
          metaBlock.appendChild(list);
          detailBody.appendChild(metaBlock);
        }
      }
      details.hidden = detailBody.children.length === 0;
    }

    function select(index) {
      if (index < 0 || index >= steps.length) return;
      stopPlayback();
      selected = index;
      showStep();
    }

    function renderStages(flow) {
      stages.replaceChildren();
      const path = flow && Array.isArray(flow.architecture) ? flow.architecture : [];
      for (const stage of path.slice(0, MAX_STAGES)) {
        if (!stage || typeof stage !== "object") continue;
        const name = limited(stage.stage || "stage", 40).replaceAll("_", " ");
        const chip = element("span", "tt__stage", name);
        const modules = Array.isArray(stage.modules) ? stage.modules.length : 0;
        if (modules) chip.appendChild(element("small", "", String(modules)));
        stages.appendChild(chip);
      }
      if (path.length > MAX_STAGES) {
        stages.appendChild(element("span", "tt__stage tt__stage--more", `+${path.length - MAX_STAGES} more`));
      }
      if (!stages.children.length) stages.appendChild(element("span", "tt__stage tt__stage--more", "—"));
    }

    function render(flow, nextQid, status = "ready") {
      if (destroyed) return;
      const identity = nextQid == null ? "" : String(nextQid);
      const identityChanged = identity !== qid;
      const previousOriginalIndex = steps[selected]?.index ?? 0;
      if (identityChanged) {
        stopPlayback();
        selected = 0;
        qid = identity;
      }
      const input = flow && typeof flow === "object" && !Array.isArray(flow) && Array.isArray(flow.steps)
        ? flow.steps : [];
      const validSteps = input.filter((step) => step && typeof step === "object" && !Array.isArray(step));
      totalSteps = validSteps.length;
      const headCount = Math.ceil(MAX_STEPS / 2);
      const tailCount = MAX_STEPS - headCount;
      steps = validSteps.length > MAX_STEPS
        ? [
            ...validSteps.slice(0, headCount).map((step, index) => ({ step, index })),
            ...validSteps.slice(-tailCount).map((step, index) => ({ step, index: totalSteps - tailCount + index })),
          ]
        : validSteps.map((step, index) => ({ step, index }));
      empty.textContent = !identity
        ? "Send a message to see its recorded steps."
        : status === "loading"
          ? "Loading recorded steps…"
          : status === "unavailable"
            ? "Trace unavailable. The engine may have restarted or the trace may have expired."
            : "No dialogue steps were recorded for this turn.";
      if (!identityChanged && steps.length) {
        const nextSelected = steps.findIndex((entry) => entry.index >= previousOriginalIndex);
        selected = nextSelected < 0 ? steps.length - 1 : nextSelected;
      }
      rail.replaceChildren();
      stepButtons = [];
      steps.forEach(({ step, index: originalIndex }, index) => {
        if (index > 0 && originalIndex > steps[index - 1].index + 1) {
          const omitted = originalIndex - steps[index - 1].index - 1;
          rail.appendChild(element("li", "tt__rail-gap", `${omitted} middle steps omitted`));
        }
        const item = element("li", "tt__rail-item");
        const button = element("button", "tt__step");
        button.type = "button";
        const stepRole = limited(step.role || "unknown", 32);
        const stepPhase = limited(step.phase || "step", 48);
        button.setAttribute("aria-label", `Step ${originalIndex + 1} of ${totalSteps}: ${stepRole}, ${stepPhase}`);
        button.appendChild(element("span", "tt__step-number", String(originalIndex + 1).padStart(2, "0")));
        button.appendChild(element("span", "tt__step-phase", stepPhase.replaceAll("_", " ")));
        button.setAttribute("data-kind", roleKind(stepRole));
        button.addEventListener("click", () => select(index));
        item.appendChild(button);
        rail.appendChild(item);
        stepButtons.push(button);
      });
      renderStages(flow);
      showStep();
    }

    previous.addEventListener("click", () => select(selected - 1));
    next.addEventListener("click", () => select(selected + 1));
    play.addEventListener("click", () => {
      if (timer != null) {
        stopPlayback();
        return;
      }
      if (reducedMotion() || steps.length < 2) return;
      if (selected === steps.length - 1) {
        selected = 0;
        showStep();
      }
      play.textContent = "Ⅱ Pause";
      play.setAttribute("aria-pressed", "true");
      timer = setInterval(() => {
        if (selected >= steps.length - 1) {
          stopPlayback();
          return;
        }
        selected++;
        showStep();
        if (selected === steps.length - 1) stopPlayback();
      }, PLAY_INTERVAL_MS);
    });

    function onMotionChange() {
      if (reducedMotion()) stopPlayback();
      showStep();
    }
    if (motion) {
      if (typeof motion.addEventListener === "function") motion.addEventListener("change", onMotionChange);
      else if (typeof motion.addListener === "function") motion.addListener(onMotionChange);
    }
    render(null, "");

    return {
      render,
      destroy() {
        if (destroyed) return;
        destroyed = true;
        stopPlayback();
        if (motion) {
          if (typeof motion.removeEventListener === "function") motion.removeEventListener("change", onMotionChange);
          else if (typeof motion.removeListener === "function") motion.removeListener(onMotionChange);
        }
        root.replaceChildren();
      },
    };
  }

  window.TraceTheater = { create };
})();
