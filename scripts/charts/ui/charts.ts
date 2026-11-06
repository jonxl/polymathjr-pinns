const regionNames = [
  "unstable_spiral",
  "stable_spiral",
  "unstable_node",
  "stable_node",
  "saddle",
  "center",
  "origin",
  "global",
];
export function facets(label: string, view: string) {
  const text = label.toLowerCase().replaceAll(" ", "_");
  const family = regionNames.find((name) => text.includes(name)) || "summary";
  const types = [
    ["local-plane|near_training_cloud", "Local error maps"],
    ["plane|heatmap", "Broad error maps"],
    ["directional.reach", "Directional reach"],
    ["shell|radial", "Radial errors"],
    ["area", "Accurate area"],
    ["training.solution|training_equations", "Training fit"],
    ["parabola", "Parabola"],
    ["transfer|probe_errors", "Transfer"],
    ["reach|margin", "Reach summaries"],
  ];
  return {
    family,
    type:
      view === "training"
        ? "Loss histories"
        : types.find(([pattern]) => new RegExp(pattern).test(text))?.[1] ||
          "Other charts",
  };
}
