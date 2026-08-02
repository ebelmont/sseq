import copy
import json
import jsonschema
import math
import re
import sys
import os
from collections import defaultdict
from jsonschema.exceptions import ValidationError
from jinja2 import Environment, FileSystemLoader

# The distance between successive x or y coordinates. Units are in pixels. This will be fixed
# throughout the html file, but zooming is implemented through a transformation matrix applied to
# the <g> element that contains the nodes, edges, and background grid.
scale = None


# Lifted/adapted from MIT-licensed https://github.com/slacy/pyssed/
class CssStyle:
    """A list of CSS styles, but stored as a dict.
    Can contain nested styles."""

    def __init__(self, *args, **kwargs):
        self._styles = {}
        for a in args:
            self.append(a)

        for name, value in kwargs.items():
            self._styles[name] = value

    def __getitem__(self, key):
        return self._styles[key]

    def keys(self):
        """Return keys of the style dict."""
        return self._styles.keys()

    def items(self):
        """Return iterable contents."""
        return self._styles.items()

    def append(self, other):
        """Append style 'other' to self."""
        self._styles = self.__add__(other)._styles

    def __add__(self, other):
        """Add self and other, and return a new style instance."""
        summed = copy.deepcopy(self)
        if isinstance(other, str):
            single = other.split(":")
            summed._styles[single[0]] = single[1]
        elif isinstance(other, dict):
            summed._styles.update(other)
        elif isinstance(other, CssStyle):
            summed._styles.update(other._styles)
        else:
            raise "Bad type for style"
        return summed

    def __repr__(self):
        return str(self._styles)

    def generate(self, parent="", indent=4):
        """Given a dict mapping CSS selectors to a dict of styles, generate a
        list of lines of CSS output."""
        subnodes = []
        stylenodes = []
        result = []

        for name, value in self.items():
            # If the sub node is a sub-style...
            if isinstance(value, dict):
                subnodes.append((name, CssStyle(value)))
            elif isinstance(value, CssStyle):
                subnodes.append((name, value))
            # Else, it's a string, and thus, a single style element
            elif (
                isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
            ):
                stylenodes.append((name, value))
            else:
                raise "Bad error"

        if stylenodes:
            result.append(parent.strip() + " {")
            for stylenode in stylenodes:
                attribute = stylenode[0].strip(" ;:")
                if isinstance(stylenode[1], str):
                    # string
                    value = stylenode[1].strip(" ;:")
                else:
                    # everything else (int or float, likely)
                    value = str(stylenode[1]) + "px"

                result.append(" " * indent + "%s: %s;" % (attribute, value))

            result.append("}")
            result.append("")  # a newline

        for subnode in subnodes:
            result += subnode[1].generate(
                parent=(parent.strip() + " " + subnode[0]).strip()
            )

        if parent == "":
            ret = "\n".join(result)
        else:
            ret = result

        return ret


# Theme palettes live in themes.json — the single source of truth shared with
# the EHP REPL (ehp_chart.rs reads the same file for its differential overlay
# palettes and index page). Add new themes there, not here.
def _load_theme_data():
    themes_path = os.path.join(os.path.dirname(__file__), "themes.json")
    with open(themes_path, "r") as f:
        return json.load(f)


_THEME_DATA = _load_theme_data()

# Cycling order shown in the chart UI.
THEME_ORDER = [t for t in _THEME_DATA["order"] if t in _THEME_DATA["themes"]]

THEME_PALETTES = {
    name: entry["palette"] for name, entry in _THEME_DATA["themes"].items()
}

# Which themes are "dark" (light text on dark background).
DARK_THEMES = {
    name for name, entry in _THEME_DATA["themes"].items() if entry.get("dark")
}

# Human-readable names shown on the chart's theme button.
THEME_LABELS = {
    name: entry.get("label", name) for name, entry in _THEME_DATA["themes"].items()
}

# Legacy aliases for the built-in palettes.
CATPPUCCIN_LATTE = THEME_PALETTES["light"]

CATPPUCCIN_MOCHA = THEME_PALETTES["dark"]
FINNJUHL_TEAK = THEME_PALETTES["teak"]
FINNJUHL_LINEN = THEME_PALETTES["linen"]
OKABE_LIGHT = THEME_PALETTES["access"]
OKABE_DARK = THEME_PALETTES["access-dark"]

global_css = CssStyle()
current_theme = "light"  # Track current theme


def get_theme_colors(theme="light"):
    """Get the appropriate color palette for the theme."""
    return THEME_PALETTES.get(theme, CATPPUCCIN_LATTE)


def get_themed_color_aliases(theme="light", fiber=False):
    """Generate color aliases using the appropriate Catppuccin theme.

    With `fiber=True`, also emit the fiber-view map colors mapE/mapH/mapP
    (no 'n' prefix — n-prefixed aliases get auto-dashed below). Every theme
    in themes.json shares the same palette role names, so picking distinct
    roles here yields distinct, readable colors in every theme. Fiber-only
    so sphere/stem chart output stays byte-identical.
    """
    colors = get_theme_colors(theme)
    aliases = {
        # Legacy color names mapped to Catppuccin
        "darkcyan": colors["teal"],      # Use teal instead of sapphire for darkcyan
        "darkgreen": colors["green"],
        "gray": colors["text"],          # Use theme text color for default nodes/edges
        "red": colors["red"],
        "blue": colors["blue"],
        "purple": colors["mauve"],
        "magenta": colors["pink"],
        "orange": colors["peach"],
        
        # Theme-specific colors
        "background": colors["base"],
        "text": colors["text"],
        "surface": colors["surface0"],
        "grid": colors["surface1"],
        
        # Differential edge types (d-types, solid lines)
        "d2": colors["teal"],            # d2 differential edges - teal
        "d3": colors["red"],             # d3 differential edges - red
        "d4": colors["green"],           # d4 differential edges - green
        "d5": colors["blue"],            # d5 differential edges - blue
        "d6": colors["yellow"],          # d6 differential edges - yellow
        "d7": colors["peach"],           # d7 differential edges - peach
        "d8": colors["mauve"],           # d8 differential edges - mauve
        
        # Nulldif edge types (n-types, dashed lines)
        "n2": colors["teal"],            # n2 nulldif edges - teal dashed
        "n3": colors["red"],             # n3 nulldif edges - red dashed
        "n4": colors["green"],           # n4 nulldif edges - green dashed
        "n5": colors["blue"],            # n5 nulldif edges - blue dashed
        "n6": colors["yellow"],          # n6 nulldif edges - yellow dashed
        "n7": colors["peach"],           # n7 nulldif edges - peach dashed
        "n8": colors["mauve"],           # n8 nulldif edges - mauve dashed

        # Note: nulldifs now use n2, n3, n4, etc. which are already defined above
    }
    if fiber:
        aliases.update(
            {
                # Fiber-sequence map colors (fiber view only)
                "mapE": colors["sapphire"],  # E: S^N -> Omega S^{N+1}
                "mapH": colors["maroon"],    # H: Omega S^{N+1} -> Omega S^{2N+1}
                "mapP": colors["green"],     # P: Omega S^{2N+1} -> S^N
            }
        )
    return aliases


def build_theme_css(initial_theme="light", fiber=False):
    """
    Emit one `:root[data-theme="X"]` CSS-variable block per theme, covering both
    the chrome variables (--bg-color etc.) and every themed color alias as
    --cc-<alias>. The generated chart classes reference these variables, so
    switching theme at runtime is a single data-theme attribute flip on <html>
    — no per-element restyling and no regeneration.

    The block for `initial_theme` also matches a bare `:root` so the chart
    renders correctly before any JS runs (flash prevention).
    """
    if initial_theme not in THEME_PALETTES:
        initial_theme = "light"
    blocks = []
    for name in THEME_ORDER:
        pal = THEME_PALETTES[name]
        theme_vars = {
            "--bg-color": pal["base"],
            "--text-color": pal["text"],
            "--surface-color": pal["surface0"],
            "--grid-color": pal["surface1"],
            "--button-bg": pal["surface1"],
            "--button-hover": pal["surface2"],
            "--button-text": pal["text"],
            "--highlight-color": pal["teal"],
            "--accent-color": pal["blue"],
            "--muted-color": pal["subtext0"],
            "--panel-bg": pal["mantle"],
        }
        for alias, color in get_themed_color_aliases(name, fiber=fiber).items():
            theme_vars[f"--cc-{alias}"] = color
        body = "\n".join(f"      {k}: {v};" for k, v in theme_vars.items())
        selector = f':root[data-theme="{name}"]'
        if name == initial_theme:
            selector = f":root, {selector}"
        blocks.append(f"    {selector} {{\n{body}\n    }}")
    return "\n".join(blocks)


def load_schema():
    schema_path = os.path.join(os.path.dirname(__file__), "input_schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    return schema


schema = load_schema()


def load_template():
    env = Environment(loader=FileSystemLoader(searchpath=os.path.dirname(__file__)))
    template = env.get_template("template.html.jinja")
    return template


def get_schema_default(data, path):
    """
    Get the default value from the schema at the given path.

    This is useful for when we need to know the default value of a field in the schema, but the field
    is not present in the data.
    """

    default_value = schema
    for key in path:
        default_value = default_value["properties"][key]
    return default_value["default"]


def get_value_or_schema_default(data, path):
    """
    Attempt to get a value from `data` at the given path.

    If it is not specified, get the default value from the schema. The schema is always assumed to
    contain a default value for the given path.
    """

    try:
        current_value = data
        for key in path:
            current_value = current_value[key]
        return current_value
    except KeyError:
        return get_schema_default(data, path)


def cssify_name(name):
    """Get a CSS class selector from a name by adding a dot prefix."""
    return "." + name


def style_and_aliases_from_attributes(attributes):
    """
    Given a list of attributes, return a `CssStyle` object that contains the union of all raw
    attribute objects, and a list of aliases.

    We return the aliases separately because we may want to specify them in a `class` attribute
    instead of a `style` attribute.
    """

    new_style = CssStyle()
    aliases = []
    for attr in attributes:
        if isinstance(attr, dict):
            # This is a raw attribute object
            for key, value in attr.items():
                if key == "color":
                    if (value_key := cssify_name(value)) in global_css.keys():
                        # This is a color alias
                        new_style += global_css[value_key]
                    else:
                        # This is a CSS color value
                        new_style += {"fill": value, "stroke": value}
                elif key == "size":
                    new_style += {"r": scale * float(value)}
                elif key == "thickness":
                    new_style += {"stroke-width": scale * float(value)}
                elif key == "fill":
                    if value == "none":
                        new_style += {"fill": "none"}
                    else:
                        new_style += {"fill": value}
                elif key == "arrowTip":
                    if value == "none":
                        new_style += {"marker-end": "none"}
                    else:
                        # We only support a few hardcoded arrow tips. To define a new arrow tip
                        # `foo`, you need to define a `<marker>` element with id `arrow-foo` in the
                        # template file. See the `arrow-simple` marker for an example.
                        new_style += {"marker-end": f"url(#arrow-{value})"}
                elif key == "pattern":
                    # We only support a few hardcoded patterns
                    if value == "solid":
                        new_style += {"stroke-dasharray": "none"}
                    elif value == "dashed":
                        new_style += {"stroke-dasharray": "5, 5"}
                    elif value == "dotted":
                        new_style += {
                            "stroke-dasharray": "0, 2",
                            "stroke-linecap": "round",
                        }
                    # Other values impossible due to schema
                elif key in ["shape", "width", "height", "visibleText"]:
                    # Skip TikZ-specific attributes - these are handled separately in generate_nodes_svg
                    pass
                else:
                    # Just treat the key-value pair as raw CSS
                    new_style += {key: value}
        elif isinstance(attr, str):
            # This is a style alias
            aliases.append(cssify_name(attr).removeprefix("."))
    return (new_style, aliases)


def generate_style(style, aliases):
    """Collapse a list of styles and aliases into a single `CssStyle` object."""
    style = copy.deepcopy(style)
    for alias in aliases:
        style.append(global_css[cssify_name(alias)])
    return style


def ensure_json_path_is_defined(data, path):
    """
    Ensure that the path exists in the JSON data, creating it if necessary.

    This modifies `data` in-place. If the path doesn't already exist, we create a JSON object, which
    is equivalent to a Python `dict`.
    """

    current_value = data
    for key in path:
        if key not in current_value:
            current_value[key] = {}
        current_value = current_value[key]


def compute_chart_dimensions(data):
    """
    This modifies `data` in-place to set up the `header.chart.width` and `header.chart.height`
    objects. Namely, it replaces the `null` values by autodetected boundaries.

    The bounds on the width and height are calculated based on the positions of the nodes in the
    chart. For maximum values, we give the smallest even size that makes the last column/row empty.
    We do the opposite for minimum values. Defaults to a 2x2 first quadrant grid if there are no
    nodes.
    """

    nodes = data.get("nodes", {})

    def compute_dimension_bounds(dim_name, coord_name, default):
        ensure_json_path_is_defined(data, ["header", "chart", dim_name])
        if data["header"]["chart"][dim_name].get("min") is None:
            # Greatest even number strictly smaller than the minimum coordinate of any node
            dimension = 2 * (
                min((node[coord_name] for node in nodes.values()), default=default) // 2
                - 1
            )
            data["header"]["chart"][dim_name]["min"] = dimension
        if data["header"]["chart"][dim_name].get("max") is None:
            # Smallest even number strictly greater than the maximum coordinate of any node
            dimension = 2 * (
                max((node[coord_name] for node in nodes.values()), default=default) // 2
                + 1
            )
            data["header"]["chart"][dim_name]["max"] = dimension

    # Arbitrary default values. These are only used if there are no nodes.
    compute_dimension_bounds("width", "x", 0)
    compute_dimension_bounds("height", "y", 0)


def calculate_absolute_positions(data):
    """
    Compute the final positions of the nodes in the chart.

    This modifies `data` in-place to add attributes `absoluteX` and `absoluteY`. They will be used
    by the SVG generation code to place the nodes at the correct positions and to draw the edges.
    """

    nodes_by_bidegree = defaultdict(list)

    # Group nodes by bidegree
    for node_id, node in data.get("nodes", {}).items():
        x, y = node["x"], node["y"]
        nodes_by_bidegree[x, y].append(node_id)

    # Sort bidegrees by the `position` attribute of the nodes
    default_position = schema["properties"]["nodes"]["additionalProperties"][
        "properties"
    ]["position"]["default"]
    for bidegree, nodes in nodes_by_bidegree.items():
        nodes_by_bidegree[bidegree] = sorted(
            nodes,
            key=lambda node_id: data["nodes"][node_id].get(
                "position", default_position
            ),
        )

    # Get defaults and compute constants
    node_size = get_value_or_schema_default(data, ["header", "chart", "nodeSize"])
    node_spacing = get_value_or_schema_default(data, ["header", "chart", "nodeSpacing"])
    node_slope = get_value_or_schema_default(data, ["header", "chart", "nodeSlope"])

    distance_between_centers = node_spacing + 2 * node_size

    # Calculate the angle of the line that the nodes will be placed on
    if node_slope is not None:
        theta = math.atan(node_slope)
    else:
        # null means vertical
        theta = math.pi / 2

    # Calculate absolute positions
    for (x, y), nodes in nodes_by_bidegree.items():
        bidegree_rank = len(nodes)
        first_center_to_last_center = (bidegree_rank - 1) * distance_between_centers
        for i, node_id in enumerate(nodes):
            node = data["nodes"][node_id]
            # Check if absolute coordinates are already set (for TikZ compatibility)
            if "absoluteX" not in node or "absoluteY" not in node:
                offset = -first_center_to_last_center / 2 + i * distance_between_centers
                data["nodes"][node_id]["absoluteX"] = x + offset * math.cos(theta)
                data["nodes"][node_id]["absoluteY"] = y + offset * math.sin(theta)


def generate_nodes_svg(data):
    """Generate an SVG <g> element containing all nodes."""

    nodes_svg = '<g id="nodes-group">\n'

    # Radius of a default node, used to size the stem-mode "?" uncertainty
    # glyph. Only consulted when a node carries an uncertain_* attribute
    # (stem mode only), so sphere charts are unaffected.
    default_node_radius = scale * get_value_or_schema_default(
        data, ["header", "chart", "nodeSize"]
    )

    node_view_mode = data.get("header", {}).get("metadata", {}).get("viewMode")

    for node_id, node in data.get("nodes", {}).items():
        cx = node["absoluteX"] * scale
        cy = node["absoluteY"] * scale

        attributes = node.get("attributes", [])
        style, aliases = style_and_aliases_from_attributes(attributes)
        
        # Extract shape information from attributes
        node_shape = "circle"  # default
        node_width = None
        node_height = None
        visible_text = ""
        
        for attr in attributes:
            if isinstance(attr, dict):
                if "shape" in attr:
                    node_shape = attr["shape"]
                if "width" in attr:
                    node_width = attr["width"] * scale
                if "height" in attr:
                    node_height = attr["height"] * scale
                if "visibleText" in attr:
                    visible_text = attr["visibleText"]

        style = style.generate(indent=0).replace("\n", " ").strip(" {}")
        if style:
            style = f'style="{style}"'
        aliases = " ".join(aliases)

        label = node.get("label", "")

        # Build data-jmap attribute if J-map targets exist
        jmap_attr = ""
        jmap_value = node.get("jmap")
        if jmap_value:
            if isinstance(jmap_value, list):
                jmap_str = ";".join(jmap_value)
            else:
                jmap_str = str(jmap_value)
            jmap_attr = f' data-jmap="{jmap_str}"'

        # Stem-mode "?" uncertainty markers: nodes flagged uncertain_src /
        # uncertain_tgt by jsonmaker get a data-uncertain attribute (the
        # contract with the chart's injected JS) and a question-mark glyph
        # centered on the node. jsonmaker only emits these attributes in
        # stem mode, so sphere charts are byte-identical.
        uncertain_attr = ""
        uncertain_mark = ""
        unc_src = "uncertain_src" in attributes
        unc_tgt = "uncertain_tgt" in attributes
        if unc_src or unc_tgt:
            kind = "both" if (unc_src and unc_tgt) else ("src" if unc_src else "tgt")
            uncertain_attr = f' data-uncertain="{kind}"'
            if node_view_mode == "fiber":
                # Fiber charts: sized ~4x the node radius and offset to the
                # upper right so the glyph clears the dot. Gated on fiber so
                # existing stem charts stay byte-identical.
                mark_font_size = 4.0 * default_node_radius
                mark_offset = 1.4 * default_node_radius
                uncertain_mark = (
                    f'<text class="uncertain-mark" x="{cx + mark_offset}" y="{cy - mark_offset}" '
                    f'text-anchor="start" '
                    f'dominant-baseline="central" font-size="{mark_font_size}px" '
                    f'pointer-events="none">?</text>\n'
                )
            else:
                # Stem charts (original form): 1.1x, centered on the node.
                mark_font_size = 1.1 * default_node_radius
                uncertain_mark = (
                    f'<text class="uncertain-mark" x="{cx}" y="{cy}" '
                    f'text-anchor="middle" '
                    f'dominant-baseline="central" font-size="{mark_font_size}px" '
                    f'pointer-events="none">?</text>\n'
                )

        # Generate appropriate SVG element based on shape
        if node_shape in ["square", "rectangle"]:
            # Use provided dimensions or defaults
            width = node_width or (scale * 0.09)  # default square size
            height = node_height or width

            # Position rectangle using the same Y coordinate reference as text (center-based)
            # Both rect and text will use y="{cy}" so they get identical coordinate transformations
            nodes_svg += f'<rect id="{node_id}" class="defaultNode {aliases}" x="{cx-width/2}" y="{cy-height/2}" width="{width}" height="{height}" {style} data-label="{label}"{jmap_attr}{uncertain_attr}></rect>\n'

            # Add visible text if present
            if visible_text:
                nodes_svg += f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" fill="white" font-size="8" class="node-text">{visible_text}</text>\n'
            nodes_svg += uncertain_mark
        else:
            # Default circle behavior (backward compatible)
            nodes_svg += f'<circle id="{node_id}" class="defaultNode {aliases}" cx="{cx}" cy="{cy}" {style} data-label="{label}"{jmap_attr}{uncertain_attr}></circle>\n'
            nodes_svg += uncertain_mark

    nodes_svg += "</g>\n"
    return nodes_svg


def generate_edges_svg(data):
    """Generate an SVG <g> element containing all edges."""

    edges_svg = '<g id="edges-group">\n'

    for edge in data.get("edges", []):
        source = data["nodes"][edge["source"]]
        if "target" in edge:
            target = data["nodes"][edge["target"]]
            target_x = target["absoluteX"] * scale
            target_y = target["absoluteY"] * scale
        elif "offset" in edge:
            target_x = (source["absoluteX"] + edge["offset"]["x"]) * scale
            target_y = (source["absoluteY"] + edge["offset"]["y"]) * scale
        else:
            # Impossible due to schema
            raise NotImplementedError

        x1 = source["absoluteX"] * scale
        y1 = source["absoluteY"] * scale

        attributes = edge.get("attributes", [])
        style, aliases = style_and_aliases_from_attributes(attributes)
        style = style.generate(indent=0).replace("\n", " ").strip(" {}")
        aliases = " ".join(aliases)

        # Add data-source/data-target for JS-based edge highlighting
        source_id = edge["source"]
        target_id = edge.get("target", "")
        data_attrs = f' data-source="{source_id}"'
        if target_id:
            data_attrs += f' data-target="{target_id}"'

        if edge.get("bezier"):
            control_points = edge["bezier"]
            if len(control_points) == 1:
                control_x = control_points[0]["x"] * scale + x1
                control_y = control_points[0]["y"] * scale + y1
                curve_d = f"Q {control_x} {control_y} {target_x} {target_y}"
            elif len(control_points) == 2:
                control0_x = control_points[0]["x"] * scale + x1
                control0_y = control_points[0]["y"] * scale + y1
                control1_x = control_points[1]["x"] * scale + target_x
                control1_y = control_points[1]["y"] * scale + target_y
                curve_d = f"C {control0_x} {control0_y} {control1_x} {control1_y} {target_x} {target_y}"
            else:
                # Impossible due to schema
                raise NotImplementedError
            # For paths, we only want stroke styling, not fill
            # Remove all fill properties and keep only stroke properties
            import re
            path_style = re.sub(r'fill:\s*[^;]+;?\s*', '', style)
            if not path_style.strip():
                # If no stroke properties, convert the fill color to stroke
                fill_match = re.search(r'fill:\s*([^;]+)', style)
                if fill_match:
                    path_style = f"stroke: {fill_match.group(1)};"
            edge_svg = f'<path d="M {x1} {y1} {curve_d}" class="defaultEdge {aliases}" style="fill: none;{path_style}"{data_attrs}></path>\n'
        else:
            edge_svg = f'<line x1="{x1}" y1="{y1}" x2="{target_x}" y2="{target_y}" class="defaultEdge {aliases}" style="{style}"{data_attrs}></line>\n'

        # Remove empty style attribute for cleaner output. This is not strictly necessary, but it
        # makes me feel better.
        edges_svg += edge_svg.replace(' style=""', "")

    edges_svg += "</g>\n"
    return edges_svg


def generate_fiber_decorations_svg(data):
    """Fiber-view chart decorations, drawn under edges and nodes.

    Emitted in PRE-NEGATION coordinates (y = f * scale, positive), exactly
    like edges/nodes: the template's DOMContentLoaded pass negates every y
    inside #content-group (lines via y1/y2, rects center-adjusted).

    The fiber sequence is unrolled one column per map source (see jsonmaker).
    Decorations: a light separator before each S^N (E-source) column so each
    stem's E/H/P triple reads as a block, and stable-range shading over
    columns whose intrinsic stem <= N - 2 (Freudenthal range). Returns ""
    outside fiber view, keeping other charts byte-identical.
    """
    meta = data.get("header", {}).get("metadata", {})
    if meta.get("viewMode") != "fiber":
        return ""
    chart = data.get("header", {}).get("chart", {})
    n_base = chart.get("fiber_n")
    columns = chart.get("fiberColumns")
    if n_base is None or not columns:
        return ""
    height = chart["height"]
    y_min, y_max = height["min"], height["max"]

    svg = '<g id="fiber-decorations">\n'

    # Stable-range shading: one band per column whose intrinsic stem <= N - 2.
    for col in columns:
        if col["stem"] <= n_base - 2:
            svg += (
                f'<rect class="fiber-stable-range" x="{(col["x"] - 0.5) * scale}" '
                f'y="{y_min * scale}" width="{scale}" '
                f'height="{(y_max - y_min) * scale}"></rect>\n'
            )

    # Separator before each S^N column: marks the start of a stem's
    # E -> H -> P triple.
    for col in columns:
        if col["n"] == n_base:
            x_sep = (col["x"] - 0.5) * scale
            svg += (
                f'<line class="fiber-separator" x1="{x_sep}" y1="{y_min * scale}" '
                f'x2="{x_sep}" y2="{y_max * scale}"></line>\n'
            )

    svg += "</g>\n"
    return svg


def generate_svg(data):
    # First make sure that the absolute positions are calculated
    calculate_absolute_positions(data)
    # We generate nodes after edges so that they are drawn on top
    # (fiber decorations, when present, go underneath everything)
    return generate_fiber_decorations_svg(data) + generate_edges_svg(data) + generate_nodes_svg(data)


def generate_html(data, theme="light"):
    # Generate CSS styles to be placed in <head>
    generate_css_styles(data, theme)
    # Calculate chart dimensions
    compute_chart_dimensions(data)
    # Generate SVG content
    static_svg_content = generate_svg(data)

    # Check if this is an Adams-Novikov chart to double the grid spacing and axis intervals
    is_adams_novikov = get_value_or_schema_default(data, ["header", "metadata", "adamsNovikov"])
    grid_spacing = scale * 2 if is_adams_novikov else scale
    axis_interval = 4 if is_adams_novikov else 2
    
    # Get theme colors for template
    theme_colors = get_theme_colors(theme)

    is_fiber = data.get("header", {}).get("metadata", {}).get("viewMode") == "fiber"

    template = load_template()
    html_output = template.render(
        data=data,
        spacing=grid_spacing,
        axis_interval=axis_interval,
        css_styles=global_css.generate(),
        static_svg_content=static_svg_content,
        theme=theme if theme in THEME_PALETTES else "light",
        theme_colors=theme_colors,
        theme_css=build_theme_css(theme, fiber=is_fiber),
        theme_names_json=json.dumps(THEME_ORDER),
        theme_labels_json=json.dumps(THEME_LABELS),
        theme_label=THEME_LABELS.get(theme, theme),
    )
    return html_output


def load_sidebyside_template():
    env = Environment(loader=FileSystemLoader(searchpath=os.path.dirname(__file__)))
    template = env.get_template("template_sidebyside.html.jinja")
    return template


def generate_sidebyside_html(so_json_file, sphere_json_file, output_file, theme="light", back_url=""):
    """
    Generate a side-by-side HTML file with SO chart on the left and sphere chart on the right.

    Args:
        so_json_file: Path to the SO chart JSON file
        sphere_json_file: Path to the sphere chart JSON file
        output_file: Path for the output HTML file
        theme: Theme to use ("light" or "dark")
        back_url: URL for the back button / j-key navigation
    """
    global global_css, scale

    # Load SO data
    with open(so_json_file, "r") as f:
        so_data = json.load(f)
    jsonschema.validate(instance=so_data, schema=schema)

    # Load sphere data
    with open(sphere_json_file, "r") as f:
        sphere_data = json.load(f)
    jsonschema.validate(instance=sphere_data, schema=schema)

    theme_colors = get_theme_colors(theme)

    # Process SO chart
    global_css = CssStyle()
    scale = get_value_or_schema_default(so_data, ["header", "chart", "scale"])
    generate_css_styles(so_data, theme)
    compute_chart_dimensions(so_data)
    so_svg_content = generate_svg(so_data)
    so_is_an = get_value_or_schema_default(so_data, ["header", "metadata", "adamsNovikov"])
    so_spacing = scale * 2 if so_is_an else scale
    so_axis_interval = 4 if so_is_an else 2
    so_css = global_css.generate()

    # Process sphere chart
    global_css = CssStyle()
    scale = get_value_or_schema_default(sphere_data, ["header", "chart", "scale"])
    generate_css_styles(sphere_data, theme)
    compute_chart_dimensions(sphere_data)
    sphere_svg_content = generate_svg(sphere_data)
    sphere_is_an = get_value_or_schema_default(sphere_data, ["header", "metadata", "adamsNovikov"])
    sphere_spacing = scale * 2 if sphere_is_an else scale
    sphere_axis_interval = 4 if sphere_is_an else 2
    sphere_css = global_css.generate()

    # Parse n and r from filenames for title generation
    import re as _re
    so_match = _re.search(r'SO(\d+)_E(\d+)', os.path.basename(so_json_file))
    if so_match:
        chart_n = int(so_match.group(1))
        chart_r = int(so_match.group(2))
    else:
        chart_n, chart_r = 0, 2

    # Build LaTeX titles matching the single-chart style
    so_title = f"$\\mathrm{{E}}_{{{chart_r}}}(\\mathrm{{SO}}({chart_n}))$"
    sphere_title = f"$\\mathrm{{E}}_{{{chart_r}}}(S^{{{chart_n}}})$"

    template = load_sidebyside_template()
    html_output = template.render(
        so_data=so_data,
        sphere_data=sphere_data,
        so_svg_content=so_svg_content,
        sphere_svg_content=sphere_svg_content,
        so_spacing=so_spacing,
        sphere_spacing=sphere_spacing,
        so_axis_interval=so_axis_interval,
        sphere_axis_interval=sphere_axis_interval,
        so_css_styles=so_css,
        sphere_css_styles=sphere_css,
        so_title=so_title,
        sphere_title=sphere_title,
        theme=theme if theme in THEME_PALETTES else "light",
        theme_colors=theme_colors,
        theme_css=build_theme_css(theme),
        theme_names_json=json.dumps(THEME_ORDER),
        theme_labels_json=json.dumps(THEME_LABELS),
        theme_label=THEME_LABELS.get(theme, theme),
        back_url=back_url,
    )

    with open(output_file, "w") as f:
        f.write(html_output)
    print(f"Generated side-by-side {output_file} successfully.")

    # Reset
    global_css = CssStyle()


def generate_css_styles(data, theme="light"):
    """Populate the global_css variable with CSS classes for color and attribute aliases."""
    global global_css

    # Themed aliases point at the per-theme CSS custom properties emitted by
    # build_theme_css(), so the generated classes are theme-independent and
    # runtime theme switching needs no restyling. User-defined aliases from
    # the data header keep their literal values (they override).
    is_fiber = data.get("header", {}).get("metadata", {}).get("viewMode") == "fiber"
    themed_colors = {
        alias: f"var(--cc-{alias})"
        for alias in get_themed_color_aliases(theme, fiber=is_fiber)
    }
    color_aliases = get_value_or_schema_default(data, ["header", "aliases", "colors"])
    final_color_aliases = {**themed_colors, **color_aliases}
    
    attribute_aliases = {
        "grid": get_schema_default(data, ["header", "aliases", "attributes", "grid"]),
        "defaultNode": get_schema_default(
            data, ["header", "aliases", "attributes", "defaultNode"]
        ),
        "defaultEdge": get_schema_default(
            data, ["header", "aliases", "attributes", "defaultEdge"]
        ),
    }
    user_attribute_aliases = get_value_or_schema_default(
        data, ["header", "aliases", "attributes"]
    )

    # Merge user-defined attribute aliases with the defaults
    for alias_name, attributes_list in user_attribute_aliases.items():
        current_attributes = attribute_aliases.get(alias_name, [])
        # This creates a new list instead of modifying the existing one, which would be bad. This is
        # because it could mutate a default value, which would ultimately corrupt `schema`.
        attribute_aliases[alias_name] = current_attributes + attributes_list

    # Generate CSS classes for color aliases. We do it first because we may need to reference them
    # in the attribute aliases.
    for color_name, color_value in final_color_aliases.items():
        css_styles = {"fill": color_value, "stroke": color_value}
        
        # Add dashing for n-type classes (nulldif edges)
        if color_name.startswith("n"):
            css_styles["stroke-dasharray"] = "5, 5"
            
        global_css += {
            cssify_name(color_name): css_styles
        }

    # Generate CSS class for nodes to set the appropriate size
    node_size = get_value_or_schema_default(data, ["header", "chart", "nodeSize"])
    global_css += {"circle": {"stroke-width": 0, "r": scale * node_size}}

    # Generate CSS classes for attribute aliases
    for alias_name, attributes_list in attribute_aliases.items():
        style, aliases = style_and_aliases_from_attributes(attributes_list)
        global_css += {cssify_name(alias_name): generate_style(style, aliases)}

    # Elements carry both an attribute alias and a color alias (e.g.
    # class="defaultEdge d3"). The color must win, but .defaultEdge/.defaultNode
    # are emitted later at equal specificity, so re-emit each color alias at
    # higher specificity. (This mis-cascade used to be masked by the runtime JS
    # restyle pass, which is gone now that theming is pure CSS.)
    for color_name in final_color_aliases:
        color_class = cssify_name(color_name)
        boosted = f".defaultNode{color_class}, .defaultEdge{color_class}"
        global_css += {boosted: dict(global_css[color_class])}

    # Stem-mode differential node states (see STEM_VIEW_SPEC.md): a class hit
    # by a d_r is a filled circle in the d_r color; a class supporting a d_r
    # is an "open" circle — filled with the background so it still catches
    # hover and occludes grid lines, with a thick d_r stroke (which must
    # override the default node `stroke-width: 0`). Defined after the
    # attribute aliases so they win against `defaultNode` at equal
    # specificity.
    for r_num in range(2, 9):
        c = f"var(--cc-d{r_num})"
        global_css += {
            cssify_name(f"diff_d{r_num}_filled"): {"fill": c, "stroke": c}
        }
        global_css += {
            cssify_name(f"diff_d{r_num}_open"): {
                "fill": "var(--bg-color)",
                "stroke": c,
                "stroke-width": scale * node_size * 0.4,
            }
        }

    # Stem-mode only: the "?" uncertainty glyph drawn on uncertain_src /
    # uncertain_tgt nodes. --text-color contrasts with both open (bg-filled)
    # and filled (d_r-colored) nodes in every theme; it is how all other
    # chart text (ticks, axes) is colored. Gated on viewMode so sphere
    # charts stay byte-identical.
    if data.get("header", {}).get("metadata", {}).get("viewMode") in ("stem", "fiber"):
        global_css += {
            ".uncertain-mark": {
                "fill": "var(--text-color)",
                "font-weight": "bold",
                # Halo so the glyph stays legible over grid lines and
                # adjacent nodes (paint-order draws the stroke underneath).
                "stroke": "var(--bg-color)",
                "stroke-width": "0.4px",
                "paint-order": "stroke",
            }
        }

    # Fiber-view-only styles: master-column separators, stable-range shading,
    # the S^{2N+1} window boundary, click-highlight focus, per-sub-column
    # tick labels, and the legend / diagnostics panels. Gated on viewMode so
    # sphere/stem charts stay byte-identical.
    if is_fiber:
        # Task 5: the fiber tooltip is now a multi-line HTML box (class name,
        # sphere, tridegree, per-map image). Give it room and readable
        # line spacing. Fiber only, so sphere/stem #tooltip is untouched.
        global_css += {
            "#tooltip": {
                "max-width": "22em",
                "line-height": "1.4",
                "text-align": "left",
            }
        }
        global_css += {
            ".fiber-separator": {
                "stroke": "var(--grid-color)",
                "stroke-width": "1.5px",
                "fill": "none",
            }
        }
        global_css += {
            ".fiber-stable-range": {
                "fill": "var(--surface-color)",
                "stroke": "none",
                "opacity": "0.35",
            }
        }
        global_css += {
            ".fiber-window-boundary": {
                "stroke": "var(--muted-color)",
                "stroke-width": "2px",
                "stroke-dasharray": "10, 6",
                "fill": "none",
            }
        }
        # Click-highlight: selected elements carry .fiber-focus while
        # everything else gets .faded (see the injected chart JS).
        global_css += {".fiber-focus": {"opacity": "1"}}
        # Hidden-EHP-value candidate: a node that is neither hit by nor supports
        # an EHP map AND is not involved in any uncertain Adams differential.
        # By exactness such a class shouldn't exist on an exact page, so on the
        # E-infinity page it flags an exactness failure that is NOT explained by
        # a missed differential -> a candidate hidden EHP value to inspect. The
        # fiber block in process_json tags these "fiber_hidden" (max page only).
        # Slightly enlarged with a bold pink border so they stand out.
        global_css += {
            ".fiber_hidden": {
                "r": scale * node_size * 1.5,
                "fill": "var(--text-color)",
                "stroke": "#ff6ea6",
                "stroke-width": scale * node_size * 0.55,
            }
        }
        global_css += {
            ".fiber-subtick": {
                "fill": "var(--muted-color)",
                "font-size": "9pt",
            }
        }
        global_css += {
            "#fiber-legend": {
                # Hidden until the user clicks the title (see the fiber-gated
                # click handler in the template JS). Task 3.
                "display": "none",
                "position": "absolute",
                "bottom": "20px",
                "left": "20px",
                "z-index": "10",
                "font-family": "sans-serif",
                "font-size": "13px",
                "color": "var(--text-color)",
                "background-color": "var(--panel-bg)",
                "border": "1px solid var(--grid-color)",
                "border-radius": "6px",
                "padding": "8px 12px",
                "max-width": "34em",
            }
        }
        global_css += {".fiber-legend-row": {"margin": "2px 0"}}
        global_css += {
            ".fiber-swatch": {
                "display": "inline-block",
                "width": "1.6em",
                "height": "0.35em",
                "margin-right": "0.5em",
                "vertical-align": "middle",
                "border-radius": "2px",
            }
        }
        global_css += {
            ".fiber-swatch-dashed": {
                "background": "none",
                "height": "0",
                "border-bottom": "2px dashed var(--text-color)",
            }
        }
        global_css += {
            ".fiber-swatch-shade": {
                "background-color": "var(--surface-color)",
                "height": "0.9em",
            }
        }
        global_css += {
            "#fiber-diagnostics": {
                "margin-top": "6px",
                "font-size": "12px",
            }
        }
        global_css += {
            "#fiber-diagnostics ul": {
                "max-height": "30vh",
                "overflow-y": "auto",
                "margin": "4px 0",
                "padding-left": "1.4em",
            }
        }
        global_css += {
            "#fiber-diagnostics summary": {"cursor": "pointer"}
        }

    # Add faded class for highlighting mode
    global_css += {
        ".faded": {"opacity": 0.3}
    }

    # Add dashed class for nulldif edges
    global_css += {
        ".dashed": {"stroke-dasharray": "5, 5"}
    }


def process_json(input_file, output_file, theme="light", view_mode="sphere", filter_value=None, data=None):
    """Render one chart HTML from SeqSee input JSON.

    `data` (optional): an already-parsed input dict — the in-process fast path
    used by ehp_batch.py, which gets the dict straight from
    jsonmaker.process_csv and skips re-reading the .json file it just wrote
    (the file is still written for the other consumers: mapview jmap
    annotation, sidebyside targets). The dict is mutated in place below
    (metadata, stem reflection, fiber positions), which is safe because
    process_csv builds a fresh dict per chart AFTER dumping it to disk.
    Schema validation of the dict path is opt-in via SEQSEE_VALIDATE=1,
    matching jsonmaker's policy (the generator is deterministic); the
    file path keeps its unconditional validation for external callers.
    """
    global global_css

    if data is None:
        # Load input JSON
        with open(input_file, "r") as f:
            data = json.load(f)

        # validate against schema
        try:
            jsonschema.validate(instance=data, schema=schema)
        except ValidationError as e:
            print("Input JSON validation error:")
            print(e)
            sys.exit(1)
    elif os.environ.get("SEQSEE_VALIDATE"):
        try:
            jsonschema.validate(instance=data, schema=schema)
        except ValidationError as e:
            print("Input JSON validation error:")
            print(e)
            sys.exit(1)

    global scale
    scale = get_value_or_schema_default(data, ["header", "chart", "scale"])

    # Add view mode and filter info to data for template access
    if "header" not in data:
        data["header"] = {}
    if "metadata" not in data["header"]:
        data["header"]["metadata"] = {}
    
    data["header"]["metadata"]["viewMode"] = view_mode
    data["header"]["metadata"]["filterValue"] = filter_value

    # Stem mode only: reflect the x-axis so n increases right-to-left and the
    # stable range sits on the left edge. Chart bounds are computed first
    # (compute_chart_dimensions is idempotent — the later call inside
    # generate_html is then a no-op), every node's x is replaced by
    # (x_min + x_max) - x, and header.chart.x_reflect_sum tells the template
    # to print reflected tick labels. Everything else (edges, absolute
    # positions, the "?" glyphs) is positioned from node coordinates, so it
    # follows automatically. Sphere charts never enter this branch.
    if view_mode == "stem":
        compute_chart_dimensions(data)
        chart_width = data["header"]["chart"]["width"]
        x_reflect_sum = chart_width["min"] + chart_width["max"]
        for node in data.get("nodes", {}).values():
            node["x"] = x_reflect_sum - node["x"]
        data["header"]["chart"]["x_reflect_sum"] = x_reflect_sum
        # Stem charts line up same-bidegree classes DIAGONALLY (−45°, slope
        # −1) instead of the schema-default horizontal (nodeSlope 0). An
        # explicit nodeSlope in the input JSON still wins.
        data["header"]["chart"].setdefault("nodeSlope", -1)

    # Fiber mode only: jsonmaker already emits the integer unrolled-sequence
    # column as node x (S^N/S^{N+1}/S^{2N+1} each in their own full column,
    # one map step apart), so there is no fractional sub-column rewrite. We
    # only record per-column descriptors (sphere + intrinsic stem) for the
    # decorations and the custom sphere/stem tick labels, plus N.
    if view_mode == "fiber" and filter_value is not None:
        import re as _re

        # Ensure header.chart (width/height bounds) exists; idempotent, so the
        # later call inside generate_html is a no-op.
        compute_chart_dimensions(data)
        n_base = filter_value
        columns = {}
        for node_id, node in data.get("nodes", {}).items():
            m = _re.match(r"^S(\d+)_(\-?\d+)_", node_id)
            if not m:
                continue
            n_val, stem = int(m.group(1)), int(m.group(2))
            columns[node["x"]] = {"x": node["x"], "n": n_val, "stem": stem}
        # Same-cell classes stack diagonally, like stem charts.
        data["header"]["chart"].setdefault("nodeSlope", 1)
        data["header"]["chart"]["fiber_n"] = n_base
        data["header"]["chart"]["fiberColumns"] = [
            columns[x] for x in sorted(columns)
        ]

        # Task 6: within a cell, order nodes so map-TARGETS (hit; ker of the
        # outgoing map by exactness) sit at the SW/left end and map-SOURCES
        # (support an outgoing map) at the NE/right end, cutting edge
        # crossings. A SOLID edge is one that is NOT a dashed 'zero/hidden'
        # stub: it lands on a real target (has a 'target') or is a solid
        # offset arrow (its attributes carry an arrowTip dict). We overwrite
        # any existing position (fiber nodes only ever get position 0 from the
        # CSV 'shift'). calculate_absolute_positions sorts ASCENDING, so
        # position -1 -> SW/left (hit), +1 -> NE/right (support).
        def _edge_is_solid(edge):
            for a in edge.get("attributes", []):
                if a == "dashed":
                    return False
            return True

        def _edge_is_source_side(edge):
            # A node "supports" an outgoing map if it is the source of a solid
            # edge that either lands on a real target or is a solid offset
            # arrow (arrowTip). A dashed stub does not count.
            if not _edge_is_solid(edge):
                return False
            if edge.get("target"):
                return True
            for a in edge.get("attributes", []):
                if isinstance(a, dict) and "arrowTip" in a:
                    return True
            return False

        has_incoming = set()
        has_outgoing = set()
        for edge in data.get("edges", []):
            if not _edge_is_solid(edge):
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            if tgt:
                has_incoming.add(tgt)
            if src is not None and _edge_is_source_side(edge):
                has_outgoing.add(src)
        # Hidden-EHP-value candidates (max page only): a node neither hit by nor
        # supporting an EHP map, AND not involved in any uncertain Adams
        # differential (no diff_d / uncertain_* attribute), AT a tridegree where
        # exactness GENUINELY fails. The last clause is essential: a class high
        # enough in stem that its EHP maps weren't computed is "stranded" only by
        # truncation, not a real hidden value. So we require the node's tridegree
        # (sphere, stem, f) to be a flagged cell of the exactness diagnostics,
        # which are frontier-guarded (a comparison whose map source is beyond the
        # recorded data is skipped). Tag "fiber_hidden" (enlarged + pink border).
        meta = data.get("header", {}).get("metadata", {})
        fiber_max_page = bool(meta.get("fiberMaxPage"))
        flagged_cells = set()
        for c in meta.get("fiberDiagnostics", {}).get("cells", []):
            try:
                flagged_cells.add((int(c["sphere"]), int(c["stem"]), int(c["f"])))
            except (KeyError, ValueError, TypeError):
                continue
        _tri = re.compile(r"^S(\d+)_(-?\d+)_(-?\d+)")
        for node_id, node in data.get("nodes", {}).items():
            node["position"] = int(node_id in has_outgoing) - int(
                node_id in has_incoming
            )
            if (
                fiber_max_page
                and flagged_cells
                and node_id not in has_incoming
                and node_id not in has_outgoing
            ):
                m = _tri.match(node_id)
                tri = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
                attrs = node.get("attributes", [])
                involved = any(
                    isinstance(a, str)
                    and (a.startswith("diff_d") or a.startswith("uncertain_"))
                    for a in attrs
                )
                if tri in flagged_cells and not involved:
                    node.setdefault("attributes", []).append("fiber_hidden")

    # Generate HTML
    html_content = generate_html(data, theme)

    # Write to output file
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"Generated {output_file} successfully.")

    # Reset global_css for the next file
    global_css = CssStyle()


def load_multiple_charts(chart_files, theme="light"):
    """
    Load and process multiple JSON chart files for multi-chart HTML generation.
    
    Args:
        chart_files: List of JSON file paths
        theme: Theme to use for processing ("light" or "dark")
    
    Returns:
        Dict mapping chart_id -> processed chart data with SVG content
    """
    charts = {}
    
    for file_path in chart_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Validate against schema
        try:
            jsonschema.validate(instance=data, schema=schema)
        except ValidationError as e:
            print(f"Validation error in {file_path}: {e}")
            continue
            
        # Extract chart identifier from filename (e.g., S3_E2.json -> S3_E2)
        chart_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Process the chart data
        global scale
        scale = get_value_or_schema_default(data, ["header", "chart", "scale"])
        
        # Add view mode and filter info to data for template access
        if "header" not in data:
            data["header"] = {}
        if "metadata" not in data["header"]:
            data["header"]["metadata"] = {}
        
        # Generate CSS styles and SVG content
        generate_css_styles(data, theme)
        compute_chart_dimensions(data)
        svg_content = generate_svg(data)
        
        # Calculate bounds for layout
        nodes = data.get("nodes", {})
        if nodes:
            x_coords = [node["x"] for node in nodes.values()]
            y_coords = [node["y"] for node in nodes.values()]
            bounds = {
                'minX': min(x_coords) if x_coords else 0,
                'maxX': max(x_coords) if x_coords else 10,
                'minY': min(y_coords) if y_coords else 0,
                'maxY': max(y_coords) if y_coords else 10
            }
        else:
            bounds = {'minX': 0, 'maxX': 10, 'minY': 0, 'maxY': 10}
        
        # Check if this is an Adams-Novikov chart
        is_adams_novikov = get_value_or_schema_default(data, ["header", "metadata", "adamsNovikov"])
        grid_spacing = scale * 2 if is_adams_novikov else scale
        axis_interval = 4 if is_adams_novikov else 2
        
        # Store processed chart data
        charts[chart_id] = {
            'data': data,
            'svg': svg_content,
            'spacing': grid_spacing,
            'axis_interval': axis_interval,
            'bounds': bounds,
            'is_adams_novikov': is_adams_novikov
        }
        
    return charts


def create_multi_chart_template():
    """
    Create enhanced template for multi-chart navigation.
    Based on template.html.jinja but with embedded chart switching.
    """
    template_content = '''<!-- Multi-chart template for SeqSee -->
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <title>{{ title or "SeqSee Multi-Chart Navigator" }}</title>
  
  <!-- Immediate theme colors to prevent any flashing -->
  <style>
    body { 
      background-color: {{ theme_colors.base }} !important; 
      color: {{ theme_colors.text }} !important; 
    }
    svg, svg * { 
      background-color: {{ theme_colors.base }} !important; 
    }
    .defaultNode { 
      fill: {{ theme_colors.text }} !important; 
      stroke: {{ theme_colors.text }} !important; 
    }
    .defaultEdge { 
      stroke: {{ theme_colors.text }} !important; 
    }
    .grid-line { 
      stroke: {{ theme_colors.surface1 }} !important; 
    }
  </style>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.2/dist/katex.min.css" crossorigin="anonymous" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@master/fonts.css" />
  <!-- KaTeX for math rendering -->
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.2/dist/katex.min.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.2/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
  <!-- svg-pan-zoom for obvious reasons -->
  <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
  <!-- Hammer.js for touch controls -->
  <script src="https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min.js"></script>
  <!-- Path data polyfill for SVG path manipulation -->
  <script src="https://cdn.jsdelivr.net/npm/path-data-polyfill@1.0.6/path-data-polyfill.min.js"></script>
  <style>
    :root {
      --bg-color: {{ theme_colors.base }};
      --text-color: {{ theme_colors.text }};
      --surface-color: {{ theme_colors.surface0 }};
      --grid-color: {{ theme_colors.surface1 }};
      --button-bg: {{ theme_colors.surface1 }};
      --button-hover: {{ theme_colors.surface2 }};
      --button-text: {{ theme_colors.text }};
    }

    body {
      font-family: "Computer Modern Serif", serif;
      font-size: 20pt;
      margin: 0;
      overflow: hidden;
      background-color: var(--bg-color);
      color: var(--text-color);
      transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* KaTeX font size fix */
    .katex {
      font-size: 1em !important;
    }

    #canvas-container {
      width: 100vw;
      height: 100vh;
      position: absolute;
    }

/* Dynamically generated CSS styles */
{{ css_styles }}

    #tooltip {
      position: absolute;
      display: none;
      pointer-events: none;
      background-color: var(--surface-color);
      border: 1px solid var(--text-color);
      color: var(--text-color);
      padding: 5px;
      z-index: 10;
      border-radius: 4px;
    }

    #controls-container {
      position: absolute;
      top: 20px;
      right: 20px;
      z-index: 10;
      display: flex;
      gap: 10px;
      pointer-events: auto;
    }

    .control-button {
      background-color: var(--button-bg);
      color: var(--button-text);
      border: 1px solid var(--text-color);
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      transition: background-color 0.3s ease, border-color 0.3s ease;
      user-select: none;
    }

    .control-button:hover {
      background-color: var(--button-hover);
    }

    .control-button.active {
      background-color: var(--text-color);
      color: var(--bg-color);
    }

    #title-container {
      position: absolute;
      top: 20px;
      left: 20px;
      right: 20px;
      z-index: 5;
      pointer-events: none;
      text-align: center;
    }

    #main-title {
      font-size: 24px;
      font-weight: bold;
      color: var(--text-color);
      background-color: var(--bg-color);
      border: 1px solid var(--surface-color);
      padding: 15px 25px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
      display: inline-block;
      max-width: 80%;
      line-height: 1.3;
      transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }

    #status-display {
      position: absolute;
      bottom: 20px;
      left: 20px;
      z-index: 10;
      font-size: 12px;
      color: var(--text-color);
      opacity: 0.7;
      background-color: var(--bg-color);
      padding: 5px 10px;
      border-radius: 4px;
      border: 1px solid var(--surface-color);
    }

    .axis {
      stroke: var(--text-color);
      stroke-width: 2px;
    }

    .tick {
      font-size: 12pt;
      fill: var(--text-color);
    }

    .x-tick {
      text-anchor: middle;
      dominant-baseline: hanging;
    }

    .y-tick {
      text-anchor: end;
      dominant-baseline: middle;
    }
  </style>
</head>

<!-- We set `visibility: visible` in JS after some preprocessing -->
<body style="visibility: hidden; background-color: {{ theme_colors.base }}; color: {{ theme_colors.text }}">
  <div id="controls-container">
    <button class="control-button" id="theme-toggle" onclick="toggleTheme()">
      {{ "Dark" if theme == "light" else "Light" }}
    </button>
    <button class="control-button" id="chart-info" onclick="showChartInfo()">
      📊 Charts: {{ chart_count }}
    </button>
  </div>
  <div id="title-container">
    <div id="main-title"></div>
  </div>
  <div id="status-display">
    Use WASD to navigate charts • {{ available_charts }} • No page reloads!
  </div>
  <div id="canvas-container">
    <svg id="svg-canvas" width="100%" height="100%">
      <defs>
        <!-- Define the arrowhead markers -->
        <marker id='arrow-simple' orient="auto" markerWidth='3' markerHeight='4' refX='0.1' refY='2' fill="context-fill"
          stroke="context-stroke">
          <path d='M0,0 V4 L2,2 Z' />
        </marker>
        <!-- Define the grid pattern -->
        <pattern id="grid" width="120" height="120" patternUnits="userSpaceOnUse">
          <path id="grid-path" d="M 120 0 L 0 0 0 120" class="grid" style="fill: none;"/>
        </pattern>
      </defs>
      <g id="content-group" class="svg-pan-zoom_viewport">
        <!-- Translate the grid so that it covers the appropriate region outside the first quadrant -->
        <g id="origin-translate">
          <!-- Apply the grid pattern to a background rectangle -->
          <rect
            id="grid-background"
            width="2000px"
            height="2000px"
            fill="url(#grid)"
          />
        </g>
        <!-- Nodes and Edges will be dynamically loaded here -->
        <g id="dynamic-content"></g>
      </g>
      <g id="axes-group">
        <!-- X-axis -->
        <line id="x-axis" class="axis" />
        <!-- Y-axis -->
        <line id="y-axis" class="axis" />
        <!-- Blocks under and to the left to hide the content -->
        <rect id="x-block" x="0" y="0" fill="var(--bg-color)" />
        <rect id="y-block" x="0" y="0" fill="var(--bg-color)" />
        <!-- Tick marks -->
        <g id="ticks" class="tick">
          <g id="x-ticks" class="x-tick"></g>
          <g id="y-ticks" class="y-tick"></g>
        </g>
      </g>
    </svg>
  </div>
  <div id="tooltip"></div>
  <script>
    // Embedded chart datasets
    const CHART_DATA = {{ chart_data_json | safe }};
    const CHART_IDS = {{ chart_ids_json | safe }};
    
    // Theme data for toggling
    const themes = {
      light: {
        base: "#eff1f5", text: "#4c4f69", surface0: "#ccd0da", surface1: "#bcc0cc", surface2: "#acb0be",
        sapphire: "#209fb5", teal: "#179299", green: "#40a02b", yellow: "#df8e1d", red: "#d20f39", 
        blue: "#1e66f5", mauve: "#8839ef", pink: "#ea76cb", peach: "#fe640b"
      },
      dark: {
        base: "#1e1e2e", text: "#cdd6f4", surface0: "#313244", surface1: "#45475a", surface2: "#585b70",
        sapphire: "#74c7ec", teal: "#94e2d5", green: "#a6e3a1", yellow: "#f9e2af", red: "#f38ba8", 
        blue: "#89b4fa", mauve: "#cba6f7", pink: "#f5c2e7", peach: "#fab387"
      }
    };
    
    let currentTheme = sessionStorage.getItem('seqsee-theme') || "{{ theme }}";
    let currentChartId = sessionStorage.getItem('seqsee-current-chart') || "{{ default_chart }}";
    
    function parseChartId(chartId) {
      // Parse chart ID like "S3_E2" -> {n: 3, r: 2}
      const match = chartId.match(/S(\\d+)_E(\\d+)/);
      if (match) {
        return { n: parseInt(match[1]), r: parseInt(match[2]) };
      }
      return { n: 3, r: 2 }; // Default
    }
    
    function buildChartId(n, r) {
      return `S${n}_E${r}`;
    }
    
    function updateTitle() {
      const { n, r } = parseChartId(currentChartId);
      const titleText = `$\\mathrm{E}_{${r}}(S^{${n}})$`;
      document.getElementById('main-title').textContent = titleText;
      
      // Render LaTeX in the main title
      window.renderMathInElement(document.getElementById('main-title'), {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
          {left: '\\\\(', right: '\\\\)', display: false},
          {left: '\\\\[', right: '\\\\]', display: true}
        ],
        throwOnError: false
      });
    }
    
    function loadChart(chartId) {
      if (!CHART_DATA[chartId]) {
        console.error(`Chart ${chartId} not found`);
        return false;
      }
      
      console.log(`🔄 Loading chart ${chartId}`);
      
      const chartData = CHART_DATA[chartId];
      
      // Clear existing content
      const dynamicContent = document.getElementById('dynamic-content');
      dynamicContent.innerHTML = '';
      
      // Insert new chart SVG content
      dynamicContent.innerHTML = chartData.svg;
      
      // Update grid and layout
      updateChartLayout(chartData);
      
      // Update title
      updateTitle();
      
      // Apply theme colors to new content
      updateThemeColors();
      
      // Store current chart
      sessionStorage.setItem('seqsee-current-chart', chartId);
      currentChartId = chartId;
      
      // Restore viewport if available
      setTimeout(() => {
        restoreViewport();
      }, 100);
      
      console.log(`✅ Chart ${chartId} loaded successfully`);
      return true;
    }
    
    function updateChartLayout(chartData) {
      const spacing = chartData.spacing || 120;
      const bounds = chartData.bounds || { minX: 0, maxX: 10, minY: 0, maxY: 10 };
      
      // Update grid pattern
      const gridPattern = document.getElementById('grid');
      gridPattern.setAttribute('width', spacing * 2);
      gridPattern.setAttribute('height', spacing * 2);
      
      const gridPath = document.getElementById('grid-path');
      gridPath.setAttribute('d', `M ${spacing * 2} 0 L 0 0 0 ${spacing * 2}`);
      
      // Update origin translate for grid
      const originTranslate = document.getElementById('origin-translate');
      originTranslate.setAttribute('transform', `translate(${bounds.minX * spacing} ${-bounds.minY * spacing})`);
      
      // Update grid background size
      const gridBg = document.getElementById('grid-background');
      gridBg.setAttribute('width', `${(bounds.maxX - bounds.minX) * spacing}px`);
      gridBg.setAttribute('height', `${(bounds.maxY - bounds.minY) * spacing}px`);
      
      // Update ticks
      updateTicks(bounds, spacing, chartData.axis_interval || 2);
    }
    
    function updateTicks(bounds, spacing, axisInterval) {
      const xTicks = document.getElementById('x-ticks');
      const yTicks = document.getElementById('y-ticks');
      
      // Clear existing ticks
      xTicks.innerHTML = '';
      yTicks.innerHTML = '';
      
      // Generate x-ticks
      for (let i = bounds.minX; i <= bounds.maxX; i += axisInterval) {
        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        tick.setAttribute('x', i * spacing);
        tick.setAttribute('y', 0.5 * spacing);
        tick.textContent = i;
        xTicks.appendChild(tick);
      }
      
      // Generate y-ticks
      for (let j = bounds.minY; j <= bounds.maxY; j += axisInterval) {
        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        tick.setAttribute('x', 0.5 * spacing);
        tick.setAttribute('y', j * spacing);
        tick.textContent = j;
        yTicks.appendChild(tick);
      }
    }
    
    function navigateToSphere(delta) {
      const { n, r } = parseChartId(currentChartId);
      const newN = Math.max(2, Math.min(72, n + delta));
      const newChartId = buildChartId(newN, r);
      
      if (newChartId !== currentChartId && CHART_DATA[newChartId]) {
        storeViewport(); // Store current viewport before navigating
        loadChart(newChartId);
      }
    }
    
    function navigateToPage(delta) {
      const { n, r } = parseChartId(currentChartId);
      const newR = Math.max(2, Math.min(10, r + delta));
      const newChartId = buildChartId(n, newR);
      
      if (newChartId !== currentChartId && CHART_DATA[newChartId]) {
        storeViewport(); // Store current viewport before navigating
        loadChart(newChartId);
      }
    }
    
    function showChartInfo() {
      const info = `Available charts: ${CHART_IDS.length}\\nCurrent: ${currentChartId}\\nUse WASD to navigate`;
      alert(info);
    }
    
    // Include all the theme, viewport, and utility functions from template.html.jinja
    {{ theme_and_navigation_functions }}
    
    // Keyboard navigation
    window.addEventListener("keydown", function (event) {
      const panSpeed = 120; // Use standard spacing for pan speed
      switch (event.key) {
        // Add arrow key controls for panning
        case "ArrowUp":
          event.preventDefault();
          window.panZoom.panBy({ x: 0, y: panSpeed });
          break;
        case "ArrowDown":
          event.preventDefault();
          window.panZoom.panBy({ x: 0, y: -panSpeed });
          break;
        case "ArrowLeft":
          event.preventDefault();
          window.panZoom.panBy({ x: panSpeed, y: 0 });
          break;
        case "ArrowRight":
          event.preventDefault();
          window.panZoom.panBy({ x: -panSpeed, y: 0 });
          break;
        // Add + and - key controls for zooming
        case "+":
        case "=":
          event.preventDefault();
          window.panZoom.zoomIn();
          break;
        case "-":
        case "_":
          event.preventDefault();
          window.panZoom.zoomOut();
          break;
        // Add controls for resetting pan and zoom
        case "Backspace":
        case "0":
        case ")":
          event.preventDefault();
          window.panZoom.reset();
          window.panZoom.panBy({ x: 2 * 60, y: -2 * 60 });
          break;
        
        // Enhanced navigation controls
        case "s":
        case "S":
          event.preventDefault();
          navigateToSphere(1); // n -> n+1
          break;
        case "w":
        case "W":
          event.preventDefault();
          navigateToSphere(-1); // n -> n-1
          break;
        case "d":
        case "D":
          event.preventDefault();
          navigateToPage(1); // r -> r+1
          break;
        case "a":
        case "A":
          event.preventDefault();
          navigateToPage(-1); // r -> r-1
          break;
      }
    });
    
    // Initialize when DOM is loaded
    window.addEventListener('DOMContentLoaded', () => {
      console.log('🚀 SeqSee Multi-Chart Navigator initializing...');
      console.log(`📊 Available charts: ${CHART_IDS.length}`);
      console.log(`💾 Current chart: ${currentChartId}`);
      
      // Load initial chart
      if (!loadChart(currentChartId)) {
        // Fallback to first available chart
        const firstChart = CHART_IDS[0];
        if (firstChart) {
          console.log(`⚠️ Falling back to ${firstChart}`);
          loadChart(firstChart);
        }
      }
      
      // Show the page after setup
      document.body.style.visibility = "visible";
      
      console.log('✅ SeqSee Multi-Chart Navigator ready!');
    });
  </script>
</body>
</html>'''
    
    return template_content


def generate_multi_chart_html(chart_files, output_file, theme="light", title=None):
    """
    Generate a single HTML file with multiple embedded charts using SeqSee machinery.
    
    Args:
        chart_files: List of JSON file paths to embed
        output_file: Output HTML file path
        theme: Theme to use ("light" or "dark")
        title: Custom title for the navigator
    
    Returns:
        True if successful, False otherwise
    """
    print(f"🔄 Generating multi-chart HTML with {len(chart_files)} charts...")
    
    # Load and process all charts
    charts = load_multiple_charts(chart_files, theme)
    if not charts:
        print("❌ No valid charts found!")
        return False
    
    print(f"✅ Processed {len(charts)} charts: {list(charts.keys())}")
    
    # Get theme colors
    theme_colors = get_theme_colors(theme)
    
    # Generate unified CSS styles (use global_css from last chart processing)
    css_styles = global_css.generate()
    
    # Prepare chart data for JavaScript embedding
    chart_data = {}
    for chart_id, chart_info in charts.items():
        chart_data[chart_id] = {
            'svg': chart_info['svg'],
            'spacing': chart_info['spacing'],
            'axis_interval': chart_info['axis_interval'],
            'bounds': chart_info['bounds'],
            'is_adams_novikov': chart_info['is_adams_novikov']
        }
    
    chart_ids = sorted(charts.keys())
    default_chart = chart_ids[0] if chart_ids else "S3_E2"
    
    # Extract theme and navigation functions from existing template
    theme_functions = """
    function toggleTheme() {
      currentTheme = currentTheme === "light" ? "dark" : "light";
      sessionStorage.setItem('seqsee-theme', currentTheme);
      updateThemeColors();
      const button = document.getElementById("theme-toggle");
      button.textContent = currentTheme === "light" ? "Dark" : "Light";
    }

    function updateThemeColors() {
      const colors = themes[currentTheme];
      const root = document.documentElement;
      
      // Update CSS variables
      root.style.setProperty('--bg-color', colors.base);
      root.style.setProperty('--text-color', colors.text);
      root.style.setProperty('--surface-color', colors.surface0);
      root.style.setProperty('--grid-color', colors.surface1);
      root.style.setProperty('--button-bg', colors.surface1);
      root.style.setProperty('--button-hover', colors.surface2);
      root.style.setProperty('--button-text', colors.text);
      
      // Update SVG elements
      document.querySelectorAll('.axis').forEach(el => {
        el.style.stroke = colors.text;
      });
      
      document.querySelectorAll('.tick text').forEach(el => {
        el.style.fill = colors.text;
      });
      
      document.querySelectorAll('#x-block, #y-block').forEach(el => {
        el.setAttribute('fill', colors.base);
      });
      
      // Update nodes and edges
      document.querySelectorAll('.defaultNode, .gray').forEach(el => {
        if (!hasSpecificColorClass(el)) {
          el.style.setProperty('fill', colors.text, 'important');
          el.style.setProperty('stroke', colors.text, 'important');
        }
      });
      
      document.querySelectorAll('.defaultEdge').forEach(el => {
        if (!hasSpecificColorClass(el)) {
          el.style.setProperty('stroke', colors.text, 'important');
        }
      });
      
      // Update differential colors
      const colorMappings = {
        'd2': colors.teal, 'd3': colors.red, 'd4': colors.green, 'd5': colors.blue,
        'd6': colors.yellow, 'd7': colors.peach, 'd8': colors.mauve,
        'n2': colors.teal, 'n3': colors.red, 'n4': colors.green, 'n5': colors.blue,
        'n6': colors.yellow, 'n7': colors.peach, 'n8': colors.mauve
      };
      
      Object.entries(colorMappings).forEach(([className, color]) => {
        document.querySelectorAll(`.${className}`).forEach(el => {
          el.style.setProperty('fill', color, 'important');
          el.style.setProperty('stroke', color, 'important');
        });
      });
    }
    
    function hasSpecificColorClass(element) {
      return Array.from(element.classList).some(cls => 
        cls.match(/^[dn]\\d+$/) || ['dr', 'nulldif'].includes(cls)
      );
    }
    
    function getCurrentViewport() {
      if (!window.panZoom) return null;
      const pan = window.panZoom.getPan();
      const zoom = window.panZoom.getZoom();
      return { x: pan.x, y: pan.y, zoom: zoom };
    }
    
    function storeViewport() {
      const viewport = getCurrentViewport();
      if (viewport) {
        sessionStorage.setItem('seqsee_viewport', JSON.stringify(viewport));
      }
    }
    
    function restoreViewport() {
      const stored = sessionStorage.getItem('seqsee_viewport');
      if (stored && window.panZoom) {
        try {
          const viewport = JSON.parse(stored);
          setTimeout(() => {
            if (window.panZoom) {
              window.panZoom.zoom(viewport.zoom);
              window.panZoom.pan({ x: viewport.x, y: viewport.y });
            }
          }, 100);
        } catch (e) {
          console.log("Failed to restore viewport:", e);
        }
      }
    }
    
    // Initialize svg-pan-zoom
    window.panZoom = svgPanZoom("#svg-canvas", {
      zoomEnabled: true,
      panEnabled: true,
      fit: false,
      center: false,
      minZoom: 0.1,
      maxZoom: 10,
      zoomScaleSensitivity: 0.15
    });
    """
    
    # Create template and render
    template_content = create_multi_chart_template()
    env = Environment()
    template = env.from_string(template_content)
    
    # Summary of available charts for status display
    available_charts = f"E{min(int(cid.split('_E')[1]) for cid in chart_ids)}-E{max(int(cid.split('_E')[1]) for cid in chart_ids)}"
    
    html_output = template.render(
        title=title,
        theme=theme,
        theme_colors=theme_colors,
        css_styles=css_styles,
        chart_data_json=json.dumps(chart_data),
        chart_ids_json=json.dumps(chart_ids),
        default_chart=default_chart,
        chart_count=len(charts),
        available_charts=available_charts,
        theme_and_navigation_functions=theme_functions
    )
    
    # Write output
    with open(output_file, 'w') as f:
        f.write(html_output)
    
    print(f"🎉 Generated {output_file} with {len(charts)} embedded charts!")
    print(f"📊 Charts: {chart_ids}")
    print(f"🎨 Theme: {theme}")
    print(f"🚀 Ready to open - no page reloads, no color flashing!")
    
    return True


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--sidebyside":
        # Side-by-side mode: --sidebyside <so.json> <sphere.json> <output.html> [theme] [back_url]
        if len(sys.argv) < 5:
            print("Usage: seqsee --sidebyside <so.json> <sphere.json> <output.html> [theme] [back_url]")
            sys.exit(1)
        so_json = sys.argv[2]
        sphere_json = sys.argv[3]
        output_file = sys.argv[4]
        theme = sys.argv[5] if len(sys.argv) > 5 else "light"
        back_url = sys.argv[6] if len(sys.argv) > 6 else ""
        generate_sidebyside_html(so_json, sphere_json, output_file, theme, back_url)
        sys.exit(0)

    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: seqsee <input.json> <output.html> [theme] [view_mode] [filter_value]")
        print("       seqsee --multi <output.html> <chart1.json> [chart2.json] ... [theme]")
        print("       seqsee --sidebyside <so.json> <sphere.json> <output.html> [theme] [back_url]")
        print("  theme: 'light' or 'dark' (default: light)")
        print("  view_mode: 'sphere', 'stem' or 'fiber' (default: sphere)")
        print("  filter_value: integer to filter by (n for sphere, s for stem, base N for fiber)")
        sys.exit(1)

    if sys.argv[1] == "--multi":
        # Multi-chart mode
        if len(sys.argv) < 4:
            print("Usage: seqsee --multi <output.html> <chart1.json> [chart2.json] ... [theme]")
            sys.exit(1)
        
        output_file = sys.argv[2]
        chart_files = []
        theme = "light"
        
        # Parse arguments - everything except last arg (if it's a theme) is a chart file
        for i in range(3, len(sys.argv)):
            arg = sys.argv[i]
            if arg in THEME_PALETTES and i == len(sys.argv) - 1:
                theme = arg
            else:
                chart_files.append(arg)
        
        if not chart_files:
            print("❌ No chart files specified!")
            sys.exit(1)
        
        success = generate_multi_chart_html(chart_files, output_file, theme)
        sys.exit(0 if success else 1)
    else:
        # Single chart mode (existing functionality)
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        theme = sys.argv[3] if len(sys.argv) > 3 else "light"
        view_mode = sys.argv[4] if len(sys.argv) > 4 else "sphere"
        filter_value = int(sys.argv[5]) if len(sys.argv) > 5 else None

        process_json(input_file, output_file, theme, view_mode, filter_value)


if __name__ == "__main__":
    main()
