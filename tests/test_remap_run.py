from reteco.runs import remap_run


def test_ids_are_replaced_through_the_map_and_unmapped_ids_pass_through():
    run = {"q": [("a", 3.0), ("b", 2.0), ("c", 1.0)]}
    assert remap_run(run, {"b": "B"}) == {"q": [("a", 3.0), ("B", 2.0), ("c", 1.0)]}


def test_collapsed_ids_keep_the_best_ranked_entry_and_its_score():
    run = {"q": [("x1", 5.0), ("y", 4.0), ("x2", 3.0)]}
    assert remap_run(run, {"x1": "X", "x2": "X"}) == {"q": [("X", 5.0), ("y", 4.0)]}


def test_a_kept_id_that_is_also_a_map_target_is_not_duplicated():
    run = {"q": [("dup", 2.0), ("keep", 1.0)]}
    assert remap_run(run, {"dup": "keep"}) == {"q": [("keep", 2.0)]}
