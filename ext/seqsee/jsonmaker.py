import re
import sys

import pandas as pd
from compact_json import Formatter
from jsonschema import validate

import main
load_schema = main.load_schema

# Regular expressions for substitutions
substitutions = [
    # Matches any string starting with an underscore, and surrounds it in \overline{}
    (re.compile(r"^_(.*)$"), r"\\overline{\1}"),
    # Matches the dot operator, which is rendered as a centered dot in LaTeX
    (re.compile(r"\."), r"\\cdot"),
    # Matches the string "DD" and replaces it with "{D}". The braces are necessary if we want to
    # handle both "DD" -> "D" and "D" -> "\Delta".
    (re.compile(r"DD"), r"{D}"),
    # Matches the string "D" and replaces it with "\Delta", but only if it is not surrounded by
    # curly braces. The `(?<!...)` and `(?!...)` expressions are negative lookbehind and negative
    # lookahead, respectively. They ensure that whatever we are matching is in the right "context",
    # i.e. in this case not being in braces, but without including that context in the match. This
    # is necessary to avoid replacing the "D" in "{D}".
    (re.compile(r"(?<!{)D(?!})"), r"\\Delta"),
    # Matches the word "t" and replaces it with "\tau". This 't' character needs to be surrounded by
    # either the beginning or end of the line, or a non-word character (something that matches
    # `\W`).
    (re.compile(r"\bt\b"), r"\\tau"),
    # Matches word expressions starting with a non-empty string of latin letters and curly braces
    # (group 1) and ending with a non-empty string of numbers or commas (,) (group 2). We insert an
    # underscore between the two groups and wrap the second one in curly braces. This ensures that
    # subscripts are correctly rendered in LaTeX. NB: Caret (^) is a word separator.
    (re.compile(r"\b([a-zA-Z{}]+)([\d,]+)\b"), r"\1_{\2}"),
    # Matches terms of the form '(letters or curly braces)^(numbers)', and wraps the second group in
    # curly braces. This ensures that superscripts are correctly rendered in LaTeX.
    (re.compile(r"\b([a-zA-Z{}]+)\^(\d+)\b"), r"\1^{\2}"),
]

# Regular expression for detecting "again" suffixes. We strip anything that is whitespace followed
# by any number of non-word characters, then the word "again", and then anything else until the end
# of the line.
detect_again = re.compile(r"\s\W*again.*")

arrow_length = 0.7


def try_get_key(row, key, default=None):
    """
    Get a key from a row.

    Return a default value if the key is not present or the value is `nan`.
    Container values (e.g. a node's `attributes` list) are returned as-is:
    `pd.isna` on a list is element-wise, and truth-testing the resulting
    array raises once the list has 2+ elements (it only ever "worked" for
    the 0/1-element lists stem nodes used to have).
    """

    try:
        val = row[key]
    except KeyError:
        return default
    if isinstance(val, (list, tuple, set, dict)):
        return val
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        return val
    return val


def edge_offset(edge_type, arrow_length=1):
    if edge_type == "h0":
        offset = {"x": 0, "y": 1}
    elif edge_type == "h1":
        offset = {"x": 1, "y": 1}
    elif edge_type == "h2":
        offset = {"x": 3, "y": 1}
    elif edge_type == "dr":
        offset = {"x": -1, "y": 2}
    elif edge_type == "nulldif":
        offset = {"x": -1, "y": 2}
    elif edge_type == "E":
        # E-type edges (stem mode only). Stem charts are mirrored so that n
        # increases right-to-left; a suspension continuing past the stable
        # edge therefore points LEFT (toward the stable range).
        offset = {"x": -1, "y": 0}
    else:
        raise ValueError
    for key in offset:
        offset[key] *= arrow_length
    return offset


def parse_node_coordinates(node_name):
    """Parse node name to extract s, stem, f, index values.

    Handles both legacy names like '4_28_5_1' and prefixed names like
    'S4_28_5_1' or 'SO10_28_5_1'.  The prefix (everything before the first
    digit in the first segment) is stripped before parsing.
    """
    if not node_name or node_name == "loc":
        return None
    parts = node_name.split("_")
    if len(parts) >= 3:
        try:
            # Strip any non-digit prefix from the first part (e.g. "S2" -> "2",
            # "SO10" -> "10").  If the first part is already numeric this is a
            # no-op.
            first = re.sub(r"^[A-Za-z]+", "", parts[0])
            s = int(first)
            stem = int(parts[1])
            f = int(parts[2])
            index = int(parts[3]) if len(parts) > 3 else 0
            return {"s": s, "stem": stem, "f": f, "index": index}
        except ValueError:
            return None
    return None


def build_hit_map(df):
    """Map node name -> d_r height, for every node listed in some drtarget.

    Precomputed once so stem-mode node coloring is O(N) instead of scanning
    the whole frame per node.
    """
    hit = {}
    for _, row in df.iterrows():
        drt = row.get("drtarget")
        if not drt or str(drt) == "nan" or drt == "":
            continue
        src_name, _ = deduplicate_name(row["name"])
        sc = parse_node_coordinates(src_name)
        if not sc:
            continue
        for t in str(drt).split(";"):
            tc = parse_node_coordinates(t)
            if tc:
                hit[t] = tc["f"] - sc["f"]
    return hit


def build_uncertain_target_set(df):
    """Set of node names appearing in some row's `nulldif` list.

    `nulldif` holds the possible targets of an UNCERTAIN differential, so
    these nodes are uncertain-targets. Mirrors build_hit_map: precomputed
    once from the full frame (a stem-k chart's uncertain targets come from
    stem-(k+1) sources), stem mode only.
    """
    targets = set()
    for _, row in df.iterrows():
        nd = row.get("nulldif")
        if not nd or str(nd) == "nan" or nd == "":
            continue
        for t in str(nd).split(";"):
            t = t.strip()
            if t:
                targets.add(t)
    return targets


def extract_node_attributes(row, view_mode="sphere", hit_map=None, uncertain_targets=None):
    ret = []
    if row.get("tautorsion", []):
        torsion = int(row["tautorsion"])
        if torsion >= 4:
            ret.append("tau4plus")
        elif torsion > 0:
            ret.append(f"tau{torsion}")

    # Add differential coloring for stem mode. Supporting a differential wins
    # over being hit by one (STEM_VIEW_SPEC.md).
    if view_mode == "stem":
        node_name, _ = deduplicate_name(row["name"])
        drt = row.get("drtarget")
        if drt and str(drt) != "nan" and drt != "":
            # Node supports a differential — open circle in the d_r color
            sc = parse_node_coordinates(node_name)
            tc = parse_node_coordinates(str(drt).split(";")[0])
            if sc and tc:
                ret.append(f"diff_d{tc['f'] - sc['f']}_open")
        elif hit_map and node_name in hit_map:
            # Node is hit by a differential — filled circle in the d_r color
            ret.append(f"diff_d{hit_map[node_name]}_filled")

        # "?" uncertainty markers (stem mode only): a node whose row has a
        # nonempty nulldif may support an uncertain differential; a node
        # listed in some other row's nulldif may be hit by one.
        nd = row.get("nulldif")
        if nd and str(nd) != "nan" and str(nd).strip() != "":
            ret.append("uncertain_src")
        if uncertain_targets and node_name in uncertain_targets:
            ret.append("uncertain_tgt")

    return ret


def get_differential_info(row, all_nodes_df, node_name):
    """
    Determine if a node supports or is the target of a differential.
    Returns (diff_type, is_source) or None.
    """
    # Check if this node has a dr target (supports a differential)
    if row.get("drtarget") and str(row["drtarget"]) != "nan" and row["drtarget"] != "":
        # Parse differential height from current node coordinates
        source_coords = parse_node_coordinates(node_name)
        target_name = str(row["drtarget"]).split(";")[0]  # Handle multiple targets
        target_coords = parse_node_coordinates(target_name)
        
        if source_coords and target_coords:
            height = target_coords["f"] - source_coords["f"]
            return (height, True)  # This node supports a d{height} differential
    
    # Check if this node is the target of any dr differential
    for _, other_row in all_nodes_df.iterrows():
        if other_row.get("drtarget") and str(other_row["drtarget"]) != "nan":
            targets = str(other_row["drtarget"]).split(";")
            if node_name in targets:
                # This node is a target - compute differential height
                other_name, _ = deduplicate_name(other_row["name"])
                source_coords = parse_node_coordinates(other_name)
                target_coords = parse_node_coordinates(node_name)
                
                if source_coords and target_coords:
                    height = target_coords["f"] - source_coords["f"]
                    return (height, False)  # This node is target of d{height} differential
    
    return None


def extract_edge_attributes(row, edge_type, nodes):
    ret = []
    # Handle nulldif special case - column is named "nulldif" not "nulldiftarget"
    if edge_type == "nulldif":
        target_node = row["nulldif"]
        target_info = try_get_key(row, f"{edge_type}info")
    elif edge_type == "E":
        # EHP CSVs store suspension targets in the "E" column
        target_node = row["E"] if "E" in row else try_get_key(row, "Etarget", "")
        target_info = try_get_key(row, "Einfo")
    else:
        target_node = row[f"{edge_type}target"]
        # Info fields are sometimes missing, so we default to an empty string
        target_info = try_get_key(row, f"{edge_type}info")
    if edge_type == "dr":
        # Compute height for dr edges and assign specific differential type (d2, d3, d4, etc.)
        source_name, _ = deduplicate_name(row["name"])
        source_coords = parse_node_coordinates(source_name)
        # Handle semicolon-separated targets by using the first target for height calculation
        first_target = target_node.split(";")[0] if ";" in target_node else target_node
        target_coords = parse_node_coordinates(first_target)
        if source_coords and target_coords:
            height = target_coords["f"] - source_coords["f"]
            # Add specific differential type based on height
            if height > 0:
                ret.append(f"d{height}")  # e.g., "d2", "d3", "d4"
    elif edge_type == "E":
        ret.append("E")  # E-type edges for stem mode
        # An E edge is colored by the class it hits: inherit the d_r color of
        # the target node's differential state (STEM_VIEW_SPEC.md).
        first_target = str(target_node).split(";")[0]
        for attr in try_get_key(nodes.get(first_target, {}), "attributes", default=[]):
            if isinstance(attr, str) and attr.startswith("diff_d"):
                ret.append(attr.split("_")[1])  # "diff_d3_open" -> "d3"
    elif edge_type == "nulldif":
        # Compute length for nulldif based on Adams filtration difference
        source_name, _ = deduplicate_name(row["name"])
        source_coords = parse_node_coordinates(source_name)
        # Handle semicolon-separated targets by using the first target for length calculation
        first_target = target_node.split(";")[0] if ";" in target_node else target_node
        target_coords = parse_node_coordinates(first_target)
        if source_coords and target_coords:
            length = abs(target_coords["f"] - source_coords["f"])
            # Add length-based type (n2, n3, n4, etc.)
            if length > 0:
                ret.append(f"n{length}")  # e.g., "n2", "n3", "n4"
    elif target_node in nodes and edge_type not in ["nulldif"]:
        # Edges into a node inherit the attributes of their target, as long as they aren't
        # nulldiff edges (which get their own coloring logic)
        target_node_attributes = try_get_key(
            nodes.get(target_node, {}), "attributes", default=[]
        )
        ret.extend(target_node_attributes)
    if target_node == "loc" or target_info == "loc":
        # This edge is an arrow
        if edge_type == "h1":
            # We treat h1 towers differently because they are also always red
            ret.append("h1tower")
        else:
            ret.append({"arrowTip": "simple"})
    # Only process info fields for non-differential edges (h0, h1, h2, E)
    # For dr and nulldif edges, we compute their types directly above
    if target_info and edge_type not in ["dr", "nulldif"]:
        # The info field has other instructions for the edge. We treat them as aliases and let
        # SeqSee handle them. The only exception is "h", which we need to tag with "edge_type" so
        # the correct alias is applied.
        if isinstance(target_info, float):
            target_info = str(int(target_info))
        extra_attributes = target_info.split(" ")
        if "h" in extra_attributes:
            extra_attributes.remove("h")
            extra_attributes.append(f"h{edge_type}")
        ret.extend(extra_attributes)

    return ret


def label_from_node_name(node_name):
    """Apply substitutions to a node name to generate a label, wrapped in dollar signs for Latex."""
    label = node_name
    for pattern, replacement in substitutions:
        label = pattern.sub(replacement, label)
    if label:
        label = f"${label}$"
    return label


def deduplicate_name(name):
    """Deduplicate names by removing all variations of 'again'. We also return whether the name was
    a duplicate."""
    # I suggest we change the csv format to remove the "again" suffixes, and instead have a
    # semicolon-separated list of names for the targets of the edges. However, the input is
    # from a legacy dataset, so I don't have control over that.

    # First strip outer opening and closing parentheses if both are present. This makes it possible
    # for the regex to only match suffixes, while not affecting the rest of the name.
    if name.startswith("(") and name.endswith(")"):
        name = name[1:-1]
    if detect_again.search(name):
        return (detect_again.sub("", name), True)
    return (name, False)


def nodes_to_json(df, view_mode="sphere", filter_value=None, highlight_mode=None, highlight_targets=None):
    nodes = {}
    hit_map = build_hit_map(df) if view_mode == "stem" else None
    uncertain_targets = build_uncertain_target_set(df) if view_mode == "stem" else None
    for _, row in df.iterrows():
        # Process node information
        node_name, is_duplicate = deduplicate_name(row["name"])
        if is_duplicate:
            continue

        # Filter by view mode if filter_value is provided
        if filter_value is not None:
            if view_mode == "sphere" and "n" in row and int(row["n"]) != filter_value:
                continue
            elif view_mode == "stem" and int(row["stem"]) != filter_value:
                continue

        # Stem charts stop at the stable edge n = k+2 — stable copies beyond
        # it would just duplicate the last column (STEM_VIEW_SPEC.md).
        if view_mode == "stem" and "n" in row and int(row["n"]) > int(row["stem"]) + 2:
            continue

        # Set coordinates based on view mode
        if view_mode == "sphere":
            x = int(row["stem"])  # s value on x-axis
            y = int(row["Adams filtration"])  # f value on y-axis
        elif view_mode == "stem":
            x = int(row["n"]) if "n" in row else 0  # n value on x-axis
            y = int(row["Adams filtration"])  # f value on y-axis
        else:
            # Default to sphere mode
            x = int(row["stem"])
            y = int(row["Adams filtration"])

        label = try_get_key(row, "label", "")
        if not label:
            label = ""

        node_data = {
            "x": x,
            "y": y,
            "label": label,
        }
        if try_get_key(row, "weight", None):
            node_data["label"] += (
                f"    ({row['weight']})" if node_data["label"] else f"({row['weight']})"
            )
        if try_get_key(row, "shift", None):
            node_data["position"] = int(row["shift"])
        
        if attributes := extract_node_attributes(row, view_mode, hit_map, uncertain_targets):
            # Only add an attributes key if there are attributes to add
            node_data["attributes"] = attributes

        # Read J-map targets for SO nodes
        j_value = try_get_key(row, "J", "")
        if j_value and str(j_value).strip():
            j_str = str(j_value).strip()
            targets = [t.strip() for t in j_str.split(";") if t.strip()]
            if len(targets) == 1:
                node_data["jmap"] = targets[0]
            elif len(targets) > 1:
                node_data["jmap"] = targets

        nodes[node_name] = node_data
    return nodes


def edges_to_json(df, nodes, view_mode="sphere"):
    edges = []
    # Define which edge types to process based on view mode
    if view_mode == "sphere":
        edge_types = ["h0", "h1", "h2", "dr", "nulldif"]
    elif view_mode == "stem":
        edge_types = ["h0", "E"]
    else:
        edge_types = ["h0", "h1", "h2", "dr", "nulldif"]  # default to sphere
        
    for _, row in df.iterrows():
        node_name, _ = deduplicate_name(row["name"])
        if node_name not in nodes:
            # Row is outside the current slice; without this, out-of-slice
            # sources would emit dangling edges (e.g. stem-mode arrows).
            continue
        for edge_type in edge_types:
            # Handle nulldif special case - column is named "nulldif" not "nulldiftarget"
            if edge_type == "nulldif":
                target_col = "nulldif"
                info_col = f"{edge_type}info"
            elif edge_type == "E":
                # EHP CSVs store suspension targets in the "E" column
                target_col = "E" if "E" in row else "Etarget"
                info_col = "Einfo"
            else:
                target_col = f"{edge_type}target"
                info_col = f"{edge_type}info"
            if target_col not in row:
                # In some CSVs, the target column is missing completely
                continue
            target_value = row[target_col]
            if target_value == "":
                continue  # Skip processing if the value is NaN or empty
            if str(target_value) == "nan":
                continue

            # Removed verbose print for batch processing - uncomment for debugging
            # print(f"{node_name} targeting {str(target_value)}")
            for target_node in str(target_value).split(";"):
                target_info = try_get_key(row, info_col, "")
                edge_data = {"source": node_name}
                if pd.notna(target_node):
                    if target_node in nodes:
                        # This is a structline
                        edge_data["target"] = target_node
                    elif target_node == "loc" or target_info == "loc":
                        # This is an arrow
                        edge_data["offset"] = edge_offset(edge_type, arrow_length)
                    elif edge_type == "E" and view_mode == "stem":
                        # Suspension continues beyond the displayed range
                        # (past the stable edge): short right-arrow instead.
                        edge_data["offset"] = edge_offset(edge_type, arrow_length)
                    else:
                        print(
                            f"Invalid target node: ({target_node}) for {edge_type} on ({node_name})"
                        )
                        continue
                elif target_info == "free" or target_info == "loc":
                    # This is also an arrow, but with a different notation
                    edge_data["offset"] = edge_offset(edge_type, arrow_length)
                else:
                    # no edge to be drawn
                    continue

                if attributes := extract_edge_attributes(row, edge_type, nodes=nodes):
                    # Only add an attributes key if there are attributes to add
                    edge_data["attributes"] = attributes

                edges.append(edge_data)
        # Check for `tauextn` if we're printing an E_infinity page
        if target_node := try_get_key(row, "tauextn"):
            if target_node in nodes:
                edges.append(
                    {
                        "source": node_name,
                        "target": target_node,
                        "attributes": ["tauextn"],
                    }
                )
    return edges


def extract_highlight_targets(df, highlight_mode, source_n):
    """Extract the target nodes for highlighting based on the mode."""
    targets = set()
    
    if highlight_mode == "E-forward":
        # E column contains targets in n+1 sphere (forward: where current sphere maps TO)
        for _, row in df.iterrows():
            if int(row["n"]) == source_n:
                e_targets = try_get_key(row, "E", "")
                if e_targets and not pd.isna(e_targets):
                    # Split by comma and clean up
                    for target in str(e_targets).split(','):
                        target = target.strip()
                        if target:
                            targets.add(target)
    
    elif highlight_mode == "E-backward":
        # Find elements that map TO current sphere via E-map (backward: what maps INTO current sphere)
        for _, row in df.iterrows():
            e_targets = try_get_key(row, "E", "")
            if e_targets and not pd.isna(e_targets):
                # Split by comma and check if any target is in our target sphere
                for target in str(e_targets).split(','):
                    target = target.strip()
                    if target:
                        # Parse target to check if it's in our sphere
                        target_parts = target.split('_')
                        if len(target_parts) >= 2 and target_parts[0].isdigit():
                            target_n = int(target_parts[0])
                            if target_n == source_n:  # This element maps into our sphere
                                source_name = row["name"]
                                if pd.notna(source_name):
                                    targets.add(str(source_name).strip())
    
    elif highlight_mode == "H-forward":
        # H column contains targets in 2*n-1 sphere (forward: where current sphere maps TO)
        for _, row in df.iterrows():
            if int(row["n"]) == source_n:
                h_targets = try_get_key(row, "H", "")
                if h_targets and not pd.isna(h_targets):
                    for target in str(h_targets).split(','):
                        target = target.strip()
                        if target:
                            targets.add(target)
    
    elif highlight_mode == "H-backward":
        # Find elements that map TO current sphere via H-map (backward: what maps INTO current sphere)
        for _, row in df.iterrows():
            h_targets = try_get_key(row, "H", "")
            if h_targets and not pd.isna(h_targets):
                for target in str(h_targets).split(','):
                    target = target.strip()
                    if target:
                        # Parse target to check if it's in our sphere
                        target_parts = target.split('_')
                        if len(target_parts) >= 2 and target_parts[0].isdigit():
                            target_n = int(target_parts[0])
                            if target_n == source_n:  # This element maps into our sphere
                                source_name = row["name"]
                                if pd.notna(source_name):
                                    targets.add(str(source_name).strip())
    
    elif highlight_mode == "P-forward":
        # P column contains targets: odd n ≥ 5 → (n-1)/2 sphere
        if source_n % 2 == 1 and source_n >= 5:  # Only for odd n ≥ 5
            for _, row in df.iterrows():
                if int(row["n"]) == source_n:
                    p_targets = try_get_key(row, "P", "")
                    if p_targets and not pd.isna(p_targets):
                        for target in str(p_targets).split(','):
                            target = target.strip()
                            if target:
                                targets.add(target)
    
    elif highlight_mode == "P-backward":
        # Find elements that map TO current sphere via P-map (backward: what maps INTO current sphere)
        # P-map: odd n → (n-1)/2, so to find what maps to even source_n, look for odd n = 2*source_n + 1
        target_odd_n = 2 * source_n + 1
        for _, row in df.iterrows():
            if int(row["n"]) == target_odd_n:  # Look in the odd sphere that maps to us
                p_targets = try_get_key(row, "P", "")
                if p_targets and not pd.isna(p_targets):
                    for target in str(p_targets).split(','):
                        target = target.strip()
                        if target:
                            # Parse target to check if it's in our sphere
                            target_parts = target.split('_')
                            if len(target_parts) >= 2 and target_parts[0].isdigit():
                                target_n = int(target_parts[0])
                                if target_n == source_n:  # This element maps into our sphere
                                    source_name = row["name"]
                                    if pd.notna(source_name):
                                        targets.add(str(source_name).strip())

    elif highlight_mode == "J-forward":
        # J column contains targets in S^n sphere (forward: where SO(n) maps TO)
        for _, row in df.iterrows():
            if int(row["n"]) == source_n:
                j_targets = try_get_key(row, "J", "")
                if j_targets and not pd.isna(j_targets):
                    for target in str(j_targets).split(';'):
                        target = target.strip()
                        if target:
                            targets.add(target)

    return targets



def process_csv(input_file, output_file, view_mode="sphere", filter_value=None, highlight_mode=None, source_csv=None, quiet=False):
    # Define the JSON schema
    schema = load_schema()

    # Load CSV data
    df = pd.read_csv(input_file)

    # Build a minimal header - let main.py handle theming
    header = {}

    # Extract highlight targets if in highlight mode
    highlight_targets = None
    if highlight_mode and source_csv:
        # Load source CSV to get the mapping data
        source_df = pd.read_csv(source_csv)
        # Determine source_n based on the highlight mode and direction
        if highlight_mode == "E-forward" and filter_value:
            source_n = filter_value - 1  # Forward: coming from n-1 to n
        elif highlight_mode == "E-backward" and filter_value:
            source_n = filter_value + 1  # Backward: coming from n+1 to n
        elif highlight_mode == "H-forward" and filter_value:
            # Forward: coming from n to 2n-1, so source_n = (filter_value + 1) // 2
            source_n = (filter_value + 1) // 2
        elif highlight_mode == "H-backward" and filter_value:
            # Backward: coming from (n+1)/2 to n, so source_n = 2*filter_value - 1
            source_n = 2 * filter_value - 1
        elif highlight_mode == "P-forward" and filter_value:
            # Forward: coming from odd n ≥ 5 to (n-1)/2, so source_n = 2*filter_value + 1
            source_n = 2 * filter_value + 1
        elif highlight_mode == "P-backward" and filter_value:
            # Backward: coming from even n to 2n+1, so source_n = (filter_value - 1) / 2
            source_n = (filter_value - 1) // 2
        elif highlight_mode == "J-forward" and filter_value:
            # Forward: J-map from SO(n) to S^n, so source_n = filter_value (same n)
            source_n = filter_value
        else:
            source_n = filter_value
        
        if source_n:
            highlight_targets = extract_highlight_targets(source_df, highlight_mode, source_n)

    # Process nodes first
    nodes = nodes_to_json(df, view_mode, filter_value, highlight_mode, highlight_targets)

    # Process edges after, since they depend on nodes
    edges = edges_to_json(df, nodes, view_mode)

    # Combine the data into a single JSON object
    json_data = {
        "$schema": "https://raw.githubusercontent.com/JoeyBF/SeqSee/refs/heads/master/seqsee/input_schema.json",
        "header": header,
        "nodes": nodes,
        "edges": edges,
    }

    # Validation and output
    try:
        validate(instance=json_data, schema=schema)
        formatter = Formatter()
        formatter.indent_spaces = 2
        formatter.dump(json_data, output_file)
        if not quiet:
            print("JSON data successfully generated and validated against the schema.")
        return json_data
    except Exception as e:
        if not quiet:
            print("Validation error:", e)
        raise e


def process_csv_to_dict(input_file, view_mode="sphere", filter_value=None, highlight_mode=None, source_csv=None):
    """
    Process CSV to JSON data structure without writing to file.
    Useful for batch processing and multi-chart generation.
    """
    # Define the JSON schema
    schema = load_schema()

    # Load CSV data
    df = pd.read_csv(input_file)

    # Build a minimal header - let main.py handle theming
    header = {}

    # Extract highlight targets if in highlight mode
    highlight_targets = None
    if highlight_mode and source_csv:
        # Load source CSV to get the mapping data
        source_df = pd.read_csv(source_csv)
        # Determine source_n based on the highlight mode and direction
        if highlight_mode == "E-forward" and filter_value:
            source_n = filter_value - 1  # Forward: coming from n-1 to n
        elif highlight_mode == "E-backward" and filter_value:
            source_n = filter_value + 1  # Backward: coming from n+1 to n
        elif highlight_mode == "H-forward" and filter_value:
            # Forward: coming from n to 2n-1, so source_n = (filter_value + 1) // 2
            source_n = (filter_value + 1) // 2
        elif highlight_mode == "H-backward" and filter_value:
            # Backward: coming from (n+1)/2 to n, so source_n = 2*filter_value - 1
            source_n = 2 * filter_value - 1
        elif highlight_mode == "P-forward" and filter_value:
            # Forward: coming from odd n ≥ 5 to (n-1)/2, so source_n = 2*filter_value + 1
            source_n = 2 * filter_value + 1
        elif highlight_mode == "P-backward" and filter_value:
            # Backward: coming from even n to 2n+1, so source_n = (filter_value - 1) / 2
            source_n = (filter_value - 1) // 2
        elif highlight_mode == "J-forward" and filter_value:
            # Forward: J-map from SO(n) to S^n, so source_n = filter_value (same n)
            source_n = filter_value
        else:
            source_n = filter_value
        
        if source_n:
            highlight_targets = extract_highlight_targets(source_df, highlight_mode, source_n)

    # Process nodes first
    nodes = nodes_to_json(df, view_mode, filter_value, highlight_mode, highlight_targets)

    # Process edges after, since they depend on nodes
    edges = edges_to_json(df, nodes, view_mode)

    # Combine the data into a single JSON object
    json_data = {
        "$schema": "https://raw.githubusercontent.com/JoeyBF/SeqSee/refs/heads/master/seqsee/input_schema.json",
        "header": header,
        "nodes": nodes,
        "edges": edges,
    }

    # Validation
    try:
        validate(instance=json_data, schema=schema)
        return json_data
    except Exception as e:
        print(f"Validation error: {e}")
        raise e


def batch_process_csv(input_file, output_dir, view_modes=None, filter_ranges=None, quiet=False):
    """
    Batch process a single CSV file into multiple JSON outputs.
    
    Args:
        input_file: Path to CSV file
        output_dir: Directory to write JSON files 
        view_modes: List of view modes ["sphere", "stem"] (default: ["sphere"])
        filter_ranges: Dict with ranges for each view mode (default: sphere=2-72, stem=0-10)
        quiet: If True, suppress progress output
    
    Returns:
        List of generated file paths
    """
    import os
    from pathlib import Path
    
    if view_modes is None:
        view_modes = ["sphere"]
    
    if filter_ranges is None:
        filter_ranges = {
            "sphere": range(2, 73),  # S2 through S72
            "stem": range(0, 11)     # stem 0 through 10
        }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    input_path = Path(input_file)
    # Extract E number from filename (e.g., E2.csv -> 2)
    er_num = int(input_path.stem[1:]) if input_path.stem.startswith('E') and input_path.stem[1:].isdigit() else 2
    
    generated_files = []
    
    for view_mode in view_modes:
        if view_mode not in filter_ranges:
            continue
            
        for filter_value in filter_ranges[view_mode]:
            if view_mode == "sphere":
                output_file = output_dir / f"S{filter_value}_E{er_num}.json"
            else:
                output_file = output_dir / f"S{filter_value}_E{er_num}_{view_mode}.json"
            
            try:
                process_csv(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    view_mode=view_mode,
                    filter_value=filter_value,
                    quiet=quiet
                )
                generated_files.append(str(output_file))
                if not quiet:
                    print(f"  ✅ Generated {output_file.name}")
            except Exception as e:
                if not quiet:
                    print(f"  ❌ Failed to generate {output_file.name}: {e}")
    
    return generated_files


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 7:
        print("Usage: jsonmaker <input.csv> <output.json> [view_mode] [filter_value] [highlight_mode] [source_csv]")
        print("  view_mode: 'sphere' or 'stem' (default: sphere)")
        print("  filter_value: integer to filter by (n for sphere, s for stem)")
        print("  highlight_mode: 'E', 'H', or 'P' for highlighting (optional)")
        print("  source_csv: CSV file containing the source data for highlighting (optional)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    view_mode = sys.argv[3] if len(sys.argv) > 3 else "sphere"
    filter_value = int(sys.argv[4]) if len(sys.argv) > 4 else None
    highlight_mode = sys.argv[5] if len(sys.argv) > 5 else None
    source_csv = sys.argv[6] if len(sys.argv) > 6 else None

    process_csv(input_file, output_file, view_mode, filter_value, highlight_mode, source_csv)


if __name__ == "__main__":
    main()
