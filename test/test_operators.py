from src.galois.operators import select, project, distinct, nested_loop_join, group_by

def test_select():
    table = [{"a":1}, {"a":2}]
    assert select(table, lambda r: r["a"] == 2) == [{"a":2}]

def test_project():
    table = [{"a":1, "b":2}]
    assert project(table, ["a"]) == [{"a":1}]

def test_distinct():
    table = [{"a":1}, {"a":1}, {"a":2}]
    assert distinct(table) == [{"a":1}, {"a":2}]

def test_join():
    left = [{"id":1}, {"id":2}]
    right = [{"key":1, "v":"x"}]
    out = nested_loop_join(left, right, "id", "key")
    assert out == [{"id":1, "key":1, "v":"x"}]

def test_group_by():
    rows = [
        {"c":"A", "x":1},
        {"c":"A", "x":3},
        {"c":"B", "x":10},
    ]
    grouped = group_by(rows, ["c"], {"x":"sum"})
    
    # We check equivalence regardless of ordering
    assert {"c":"A", "x":4} in grouped
    assert {"c":"B", "x":10} in grouped
