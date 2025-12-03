from typing import List, Dict, Any, Callable, Tuple as Row

Table = List[Dict[str, Any]]

def select(table: Table, condition_fn: Callable[[Row], bool]) -> Table:
    """ 
    Filters the rows of the table based on a condition function
    """
    return [row for row in table if condition_fn(row)]

def project(table: Table, attrs: List[str]) -> Table:
    """ 
    Selects specific attributes (columns) from the table
    """
    return [{attr: row[attr] for attr in attrs if attr in row} for row in table]

def distinct(table: Table) -> Table:
    """
    Removes duplicate rows from the table.
    """
    seen = set()
    unique_table = []

    # Iterate through each row in the table
    for row in table:

        # Convert the row dictionary to a sorted tuple of items for hashing
        row_tuple = tuple(sorted(row.items()))

        # Check if this row has been seen before
        if row_tuple not in seen:
            # Add the row to the set of seen rows and the unique table
            seen.add(row_tuple)
            unique_table.append(row)
    return unique_table

def nested_loop_join(left: Table, right: Table, left_key: str, right_key: str) -> Table:
    """
    Simple equi-join: left[left_key] == right[right_key]
    """
    out = []
    for l in left:
        for r in right:
            if l.get(left_key) == r.get(right_key):
                merged = {**l, **r}
                out.append(merged)
    return out


def group_by(rows: Table, group_attrs: List[str], agg_spec: Dict[str, str]) -> Table:
    """
    group_attrs: attributes used to form groups
    agg_spec: {"column_name": "agg_function"} e.g. {"pop": "sum"}
    agg functions supported: sum, max, min, avg, count
    """

    groups = {}

    # Build groups
    for row in rows:
        key = tuple(row[attr] for attr in group_attrs)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    # Apply aggregates
    result = []
    for key, group_rows in groups.items():
        out_row = {attr: val for attr, val in zip(group_attrs, key)}
        
        for col, agg in agg_spec.items():
            values = [r[col] for r in group_rows if r[col] is not None]

            if agg == "sum":
                out_row[col] = sum(values)
            elif agg == "max":
                out_row[col] = max(values)
            elif agg == "min":
                out_row[col] = min(values)
            elif agg == "avg":
                out_row[col] = sum(values) / len(values) if values else None
            elif agg == "count":
                out_row[col] = len(values)
            else:
                raise ValueError(f"Unsupported aggregate: {agg}")

        result.append(out_row)

    return result