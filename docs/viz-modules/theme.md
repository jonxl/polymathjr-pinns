# Theme.jl

Colour-theme system for the diagnostic dashboard.

**Location:** `viz/Theme.jl`

---

## Struct

```julia
struct Theme
    name::String           # human-readable identifier
    bg_primary::String     # figure / scene background
    bg_secondary::String   # axis pane background
    bg_tertiary::String    # slider track / widget bg
    text_primary::String   # titles, labels, tick labels
    text_secondary::String # status bar, subtle text
    grid_color::String     # axis grid lines
    border_color::String   # axis spines / borders
end
```

All colour fields are hex strings (e.g. `"#171717"`).

---

## Built-In Themes

### Dark (default)

| Field | Value |
|-------|-------|
| `bg_primary` | `#171717` |
| `bg_secondary` | `#2b2b2b` |
| `bg_tertiary` | `#3a3a3a` |
| `text_primary` | `#d4d4d4` |
| `text_secondary` | `#888888` |
| `grid_color` | `#3a3a3a` |

### Light

| Field | Value |
|-------|-------|
| `bg_primary` | `#fafafa` |
| `bg_secondary` | `#ffffff` |
| `text_primary` | `#222222` |
| `text_secondary` | `#666666` |
| `grid_color` | `#cccccc` |

### High Contrast

| Field | Value |
|-------|-------|
| `bg_primary` | `#000000` |
| `bg_secondary` | `#000000` |
| `text_primary` | `#ffff00` |
| `text_secondary` | `#aaaa00` |
| `grid_color` | `#666666` |

---

## API

```julia
get_theme("dark")           # → DARK_THEME
get_theme("light")          # → LIGHT_THEME
get_theme("high_contrast")  # → HIGH_CONTRAST_THEME
get_theme("bogus")          # → DARK_THEME (fallback)

list_themes()               # → ["dark", "light", "high_contrast"]
```

---

*See also: [Viz.jl](viz.md), [NNViewer.jl](nnviewer.md)*
