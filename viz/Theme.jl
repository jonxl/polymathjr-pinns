# -------------------------------------------------------------------
# Theme system — colour schemes for the viz diagnostic dashboard.
#
# Themes control figure-level styling (background, text, grid).
# Data series colours are defined separately as module constants
# inside NNViewer.jl (matching the Python nn-viewer palette).
#
# Colour values are stored as hex strings so the Theme module stays
# dependency-free.  They are converted to Makie colour types inside
# NNViewer.jl at application time.
# -------------------------------------------------------------------

module Theme

export Theme, DARK_THEME, LIGHT_THEME, HIGH_CONTRAST_THEME, get_theme, list_themes

"""
    Theme

Colour scheme for the diagnostic dashboard figure.

**Fields**
- `name`           — human-readable theme identifier
- `bg_primary`     — figure / scene background
- `bg_secondary`   — axis pane background
- `bg_tertiary`    — slider track / widget background
- `text_primary`   — title, axis labels, tick labels
- `text_secondary` — status bar, subtle text
- `grid_color`     — axis grid lines
- `border_color`   — axis spines / borders
"""
struct Theme
    name::String
    bg_primary::String
    bg_secondary::String
    bg_tertiary::String
    text_primary::String
    text_secondary::String
    grid_color::String
    border_color::String
end

# ---------- built-in themes ----------------------------------------

const DARK_THEME = Theme(
    "dark",
    "#171717",    # bg_primary
    "#2b2b2b",    # bg_secondary
    "#3a3a3a",    # bg_tertiary
    "#d4d4d4",    # text_primary
    "#888888",    # text_secondary
    "#3a3a3a",    # grid
    "transparent",# border
)

const LIGHT_THEME = Theme(
    "light",
    "#fafafa",
    "#ffffff",
    "#e0e0e0",
    "#222222",
    "#666666",
    "#cccccc",
    "#dddddd",
)

const HIGH_CONTRAST_THEME = Theme(
    "high_contrast",
    "#000000",
    "#000000",
    "#333333",
    "#ffff00",
    "#aaaa00",
    "#666666",
    "#ffffff",
)

const _THEME_REGISTRY = Dict{String,Theme}(
    "dark"          => DARK_THEME,
    "light"         => LIGHT_THEME,
    "high_contrast" => HIGH_CONTRAST_THEME,
)

"""
    get_theme(name::String) → Theme

Look up a built-in theme by name (case-insensitive).  Falls back to
`DARK_THEME` for unknown names.
"""
function get_theme(name::String)
    name_lower = lowercase(name)
    return get(_THEME_REGISTRY, name_lower, DARK_THEME)
end

"""
    list_themes() → Vector{String}

Return the names of all registered themes.
"""
list_themes() = collect(keys(_THEME_REGISTRY))

end # module Theme
