"""Tests for the significance / uncertainty analysis module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from analysis.data_analysis import create_overall_precision_recall_summary
from analysis.significance import (
    _prf_from_sums,
    adjust_pvalues,
    bootstrap_ci,
    bootstrap_pooled_accuracy,
    bootstrap_prf,
    build_overall_table,
    build_per_assay_precision_recall_table,
    build_precision_recall_table,
    build_single_run_table,
    cluster_bootstrap_pooled,
    cluster_bootstrap_prf,
    cluster_bootstrap_prf_delta,
    collect_paired_data,
    collect_single_run_data,
    effective_sample_size,
    paired_mcnemar,
    paired_permutation,
    paired_permutation_prf,
    paired_wilcoxon,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestBootstrapCI:
    def test_constant_values(self) -> None:
        mean, lo, hi = bootstrap_ci([0.5, 0.5, 0.5, 0.5])
        assert mean == 0.5
        assert lo == 0.5
        assert hi == 0.5

    def test_ci_brackets_mean(self) -> None:
        values = [0.2, 0.4, 0.6, 0.8, 1.0]
        mean, lo, hi = bootstrap_ci(values, seed=1)
        assert abs(mean - 0.6) < 1e-9
        assert lo <= mean <= hi
        assert lo < hi

    def test_empty_returns_nan(self) -> None:
        mean, lo, hi = bootstrap_ci([])
        assert mean != mean  # nan
        assert lo != lo
        assert hi != hi

    def test_reproducible_with_seed(self) -> None:
        values = [0.1, 0.3, 0.9, 0.5, 0.7]
        assert bootstrap_ci(values, seed=42) == bootstrap_ci(values, seed=42)


class TestPairedWilcoxon:
    def test_all_improve_is_significant(self) -> None:
        pairs = [(0.5, 0.9)] * 10
        _stat, p, n = paired_wilcoxon(pairs)
        assert n == 10
        assert p < 0.05

    def test_no_difference_returns_one(self) -> None:
        pairs = [(0.6, 0.6)] * 8
        _stat, p, n = paired_wilcoxon(pairs)
        assert n == 0
        assert p == 1.0

    def test_empty_returns_one(self) -> None:
        _stat, p, n = paired_wilcoxon([])
        assert p == 1.0
        assert n == 0


class TestPairedMcnemar:
    def test_known_counts(self) -> None:
        # 9 ARMS-only correct, 1 baseline-only correct, plus ties.
        outcomes = [(False, True)] * 9 + [(True, False)] * 1 + [(True, True)] * 5 + [(False, False)] * 5
        result = paired_mcnemar(outcomes)
        assert result["c"] == 9
        assert result["b"] == 1
        assert result["n_discordant"] == 10

    def test_no_discordant_pairs(self) -> None:
        outcomes = [(True, True)] * 4 + [(False, False)] * 4
        result = paired_mcnemar(outcomes)
        assert result["n_discordant"] == 0
        assert result["pvalue"] == 1.0

    def test_strongly_lopsided_is_significant(self) -> None:
        outcomes = [(False, True)] * 50 + [(True, False)] * 2
        result = paired_mcnemar(outcomes)
        assert result["pvalue"] < 0.001


class TestPairedPermutation:
    def test_no_discordant_returns_one(self) -> None:
        # Every record balanced (d_i = 0): nothing to test.
        result = paired_permutation([(0, 0), (2, 2), (1, 1)])
        assert result["n_effective"] == 0
        assert result["pvalue"] == 1.0
        assert result["s_observed"] == 0.0

    def test_s_observed_is_net_arms_advantage(self) -> None:
        # (baseline_only, arms_only) per record -> d_i = arms_only - baseline_only.
        result = paired_permutation([(0, 3), (1, 2), (1, 1), (1, 0)])
        assert result["s_observed"] == 3.0  # (3-0)+(2-1)+(1-1)+(0-1)
        assert result["n_effective"] == 3  # the (1, 1) record has d_i = 0

    def test_strongly_one_sided_is_significant(self) -> None:
        # 20 records all favouring ARMS by a wide margin.
        result = paired_permutation([(0, 5)] * 20)
        assert result["pvalue"] < 0.001

    def test_clustering_is_not_overconfident(self) -> None:
        # 50 ARMS-wins vs 15 baseline-wins, but bunched into 10 records that each
        # lean one way (7 pro-ARMS, 3 pro-baseline). A flat field-level McNemar
        # would call this highly significant; the record-clustered permutation
        # should not, because the real signal is only "7 vs 3 records".
        discordant = [(0, 5)] * 7 + [(5, 0)] * 3
        perm = paired_permutation(discordant)
        flat = paired_mcnemar([(False, True)] * 35 + [(True, False)] * 15)
        # Flat McNemar calls it significant; the clustered permutation does not.
        assert flat["pvalue"] < 0.05
        assert perm["pvalue"] > 0.05
        assert perm["pvalue"] > flat["pvalue"]

    def test_reproducible_with_seed(self) -> None:
        discordant = [(1, 3), (0, 2), (2, 1), (1, 4)]
        assert paired_permutation(discordant, seed=7)["pvalue"] == paired_permutation(discordant, seed=7)["pvalue"]


class TestClusterBootstrapPooled:
    def test_pooled_is_field_weighted(self) -> None:
        # Record A: 1/1 correct; Record B: 1/3 correct. Pooled = 2/4 = 0.5,
        # which differs from the record-mean (1.0 + 0.333)/2 = 0.667.
        counts = [(1, 1, 1), (1, 1, 3)]
        point, lo, hi = cluster_bootstrap_pooled(counts, "baseline")
        assert abs(point - 0.5) < 1e-9
        assert lo <= point <= hi

    def test_empty_returns_nan(self) -> None:
        point, lo, hi = cluster_bootstrap_pooled([], "system")
        assert point != point  # nan


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _build_mini_data_root(root: Path) -> None:
    """Create a tiny atacseq dataset: 2 records, baseline weaker than ARMS."""
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "title", "permissible_values": []},
        ]
    }
    _write_record(root / "schemas" / "atacseq.json", schema)
    for name in ("r1.json", "r2.json"):
        gold = {"tissue": "lung", "title": "study"}
        _write_record(root / "atacseq" / "gold" / name, gold)
        # baseline gets the ontology field wrong; ARMS gets everything right.
        _write_record(
            root / "atacseq" / "output" / "gpt5mini" / "baseline" / name, {"tissue": "WRONG", "title": "study"}
        )
        _write_record(
            root / "atacseq" / "output" / "gpt5mini" / "agent-tool" / name, {"tissue": "lung", "title": "study"}
        )


class TestCollectPairedData:
    def test_collects_records_and_fields(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        data = collect_paired_data(tmp_path, "gpt5mini", "atacseq")
        # 2 records, each with 1 ontology + 1 non-ontology field.
        assert len(data.record_acc["all"]) == 2
        assert len(data.field_outcomes["ontology"]) == 2
        assert len(data.field_outcomes["non_ontology"]) == 2
        # Ontology field: baseline wrong, ARMS right.
        assert data.field_outcomes["ontology"][0] == (False, True)

    def test_missing_assay_returns_empty(self, tmp_path: Path) -> None:
        data = collect_paired_data(tmp_path, "gpt5mini", "atacseq")
        assert data.record_acc["all"] == []


class TestBuildOverallTable:
    def test_overall_table_shape_and_values(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        table = build_overall_table(tmp_path, "gpt5mini")
        assert list(table["category"]) == [
            "Ontology-constrained",
            "Non-ontology-constrained",
            "All fields",
        ]
        ont = table[table["category"] == "Ontology-constrained"].iloc[0]
        # baseline 0/2 correct, ARMS 2/2; McNemar c=2, b=0.
        assert ont["mcnemar_c"] == 2
        assert ont["mcnemar_b"] == 0
        # The record-clustered permutation column is present for every category.
        assert "perm_p" in table.columns


def _build_clustered_root(root: Path) -> None:
    """3 records sharing the same gold values, so each field forms one cluster:
    ARMS is right on tissue every time and wrong on title every time."""
    schema = {
        "children": [
            {"name": "tissue", "permissible_values": [{"type": "ontology"}]},
            {"name": "title", "permissible_values": []},
        ]
    }
    _write_record(root / "schemas" / "atacseq.json", schema)
    for name in ("r0.json", "r1.json", "r2.json"):
        _write_record(root / "atacseq" / "gold" / name, {"tissue": "lung", "title": "study"})
        _write_record(
            root / "atacseq" / "output" / "gpt5mini" / "agent-tool" / name, {"tissue": "lung", "title": "WRONG"}
        )


class TestEffectiveSampleSize:
    def test_perfect_clusters_collapse_to_cluster_count(self, tmp_path: Path) -> None:
        # Two clusters (tissue=lung, title=study), each with 3 perfectly-correlated
        # instances -> ICC = 1, so N_eff collapses to the number of clusters (2).
        _build_clustered_root(tmp_path)
        e = effective_sample_size(tmp_path, "gpt5mini", "agent-tool")
        assert e["n"] == 6
        assert e["n_clusters"] == 2
        assert e["icc"] > 0.99
        assert abs(e["n_effective"] - 2) < 1e-6

    def test_field_type_filter(self, tmp_path: Path) -> None:
        _build_clustered_root(tmp_path)
        ont = effective_sample_size(tmp_path, "gpt5mini", "agent-tool", field_type="ontology")
        assert ont["n"] == 3
        assert ont["n_clusters"] == 1  # only the tissue=lung cluster


class TestPrfFromSums:
    def test_matches_scalar_helper_elementwise(self) -> None:
        import numpy as np

        from analysis.metrics import precision_recall_f1

        tp = np.array([6.0, 0.0, 5.0, 0.0])
        fp = np.array([2.0, 0.0, 0.0, 4.0])
        fn = np.array([3.0, 7.0, 0.0, 0.0])
        precision, recall, f1 = _prf_from_sums(tp, fp, fn)
        for i in range(len(tp)):
            expected = precision_recall_f1({"TP": int(tp[i]), "FP": int(fp[i]), "FN": int(fn[i])})
            assert precision[i] == pytest.approx(expected["precision"])
            assert recall[i] == pytest.approx(expected["recall"])
            assert f1[i] == pytest.approx(expected["f1"])

    def test_zero_denominators_give_zero_not_nan(self) -> None:
        import numpy as np

        precision, recall, f1 = _prf_from_sums(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert (precision[0], recall[0], f1[0]) == (0.0, 0.0, 0.0)


class TestClusterBootstrapPrf:
    #                     base_tp, base_fp, base_fn, arms_tp, arms_fp, arms_fn
    CONFUSION = [(3, 2, 1, 5, 0, 1), (2, 3, 2, 6, 1, 0), (4, 1, 1, 5, 1, 1)]

    def test_point_estimate_matches_pooled_counts(self) -> None:
        from analysis.metrics import precision_recall_f1

        result = cluster_bootstrap_prf(self.CONFUSION, "system")
        expected = precision_recall_f1({"TP": 16, "FP": 2, "FN": 2})
        for metric in ("precision", "recall", "f1"):
            assert result[metric][0] == pytest.approx(expected[metric])

    def test_baseline_and_arms_select_different_columns(self) -> None:
        from analysis.metrics import precision_recall_f1

        baseline = cluster_bootstrap_prf(self.CONFUSION, "baseline")
        expected = precision_recall_f1({"TP": 9, "FP": 6, "FN": 4})
        assert baseline["precision"][0] == pytest.approx(expected["precision"])
        assert baseline["f1"][0] < cluster_bootstrap_prf(self.CONFUSION, "system")["f1"][0]

    def test_ci_brackets_the_point_estimate(self) -> None:
        result = cluster_bootstrap_prf(self.CONFUSION, "system", seed=7)
        for metric in ("precision", "recall", "f1"):
            point, lo, hi = result[metric]
            assert lo <= point <= hi

    def test_identical_records_give_a_zero_width_interval(self) -> None:
        confusion = [(3, 1, 1, 4, 0, 1)] * 5
        result = cluster_bootstrap_prf(confusion, "system")
        for metric in ("precision", "recall", "f1"):
            point, lo, hi = result[metric]
            assert lo == pytest.approx(point)
            assert hi == pytest.approx(point)

    def test_empty_returns_nan(self) -> None:
        result = cluster_bootstrap_prf([], "system")
        for metric in ("precision", "recall", "f1"):
            assert all(value != value for value in result[metric])  # nan

    def test_reproducible_with_seed(self) -> None:
        assert cluster_bootstrap_prf(self.CONFUSION, "system", seed=3) == cluster_bootstrap_prf(
            self.CONFUSION, "system", seed=3
        )


class TestClusterBootstrapPrfDelta:
    CONFUSION = TestClusterBootstrapPrf.CONFUSION

    def test_point_delta_is_system_minus_baseline(self) -> None:
        baseline = cluster_bootstrap_prf(self.CONFUSION, "baseline")
        arms = cluster_bootstrap_prf(self.CONFUSION, "system")
        delta = cluster_bootstrap_prf_delta(self.CONFUSION)
        for metric in ("precision", "recall", "f1"):
            assert delta[metric][0] == pytest.approx(arms[metric][0] - baseline[metric][0])

    def test_identical_runs_give_zero_difference_and_zero_width(self) -> None:
        """The pairing must cancel exactly, not merely average to zero."""
        confusion = [(3, 2, 1, 3, 2, 1), (5, 0, 2, 5, 0, 2), (1, 4, 3, 1, 4, 3)]
        delta = cluster_bootstrap_prf_delta(confusion)
        for metric in ("precision", "recall", "f1"):
            point, lo, hi = delta[metric]
            assert point == pytest.approx(0.0)
            assert lo == pytest.approx(0.0)
            assert hi == pytest.approx(0.0)

    def test_clear_improvement_excludes_zero(self) -> None:
        confusion = [(1, 5, 4, 6, 0, 0)] * 20
        delta = cluster_bootstrap_prf_delta(confusion)
        assert delta["f1"][1] > 0  # lower bound above zero

    def test_empty_returns_nan(self) -> None:
        delta = cluster_bootstrap_prf_delta([])
        assert all(value != value for value in delta["f1"])


class TestRecordConfusionCollection:
    def test_collect_paired_data_populates_record_confusion(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        data = collect_paired_data(tmp_path, "gpt5mini", "atacseq")
        assert len(data.record_confusion["all"]) == 2
        # Per record: baseline gets tissue wrong (substitution) and title right;
        # ARMS gets both right.  So baseline (tp, fp, fn) = (1, 1, 1), ARMS = (2, 0, 0).
        assert data.record_confusion["all"][0] == (1, 1, 1, 2, 0, 0)
        assert data.record_confusion["ontology"][0] == (0, 1, 1, 1, 0, 0)


class TestBuildPrecisionRecallTable:
    def test_table_shape_and_ordering(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        table = build_precision_recall_table(tmp_path, "gpt5mini")
        assert len(table) == 9  # 3 categories x 3 metrics
        assert list(table["metric"].unique()) == ["precision", "recall", "f1"]
        assert list(table["category"].unique()) == [
            "Ontology-constrained",
            "Non-ontology-constrained",
            "All fields",
        ]
        assert table["n_records"].iloc[0] == 2

    def test_arms_beats_baseline_on_every_row(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        table = build_precision_recall_table(tmp_path, "gpt5mini")
        for _, row in table.iterrows():
            assert not row["difference"].startswith("-")

    def test_missing_data_yields_dashes(self, tmp_path: Path) -> None:
        table = build_precision_recall_table(tmp_path, "gpt5mini")
        assert set(table["baseline"]) == {"-"}


class TestSingleRunIntervals:
    """Any condition gets an interval, and it agrees with the paired tables."""

    def test_matches_the_paired_arm_exactly(self, tmp_path: Path) -> None:
        """The same run measured alone and as an arm of a pair must not differ."""
        _build_mini_data_root(tmp_path)
        paired = collect_paired_data(tmp_path, "gpt5mini", "atacseq")

        for run_type, which in (("baseline", "baseline"), ("agent-tool", "system")):
            single = collect_single_run_data(tmp_path, "gpt5mini", run_type, "atacseq")
            for category in ("ontology", "non_ontology", "all"):
                assert bootstrap_pooled_accuracy(single.record_counts[category]) == cluster_bootstrap_pooled(
                    paired.record_counts[category], which
                ), (run_type, category)
                assert bootstrap_prf(single.record_confusion[category]) == cluster_bootstrap_prf(
                    paired.record_confusion[category], which
                ), (run_type, category)

    def test_point_estimate_is_the_reported_micro_number(self, tmp_path: Path) -> None:
        """The interval must qualify the number the data_analysis table already prints."""
        _build_mini_data_root(tmp_path)
        data = collect_single_run_data(tmp_path, "gpt5mini", "baseline")
        prf = bootstrap_prf(data.record_confusion["all"])
        weighted = create_overall_precision_recall_summary(str(tmp_path), "gpt5mini", "baseline")
        row = weighted[weighted["category"] == "all"].iloc[0]
        assert round(prf["precision"][0], 3) == row["precision"]
        assert round(prf["recall"][0], 3) == row["recall"]

    def test_works_for_a_condition_the_paired_tables_do_not_model(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        # A third run, which collect_paired_data has no notion of.
        for name in ("r1.json", "r2.json"):
            _write_record(
                tmp_path / "atacseq" / "output" / "gpt5mini" / "baseline-r2" / name,
                {"tissue": "lung", "title": "WRONG"},
            )
        table = build_single_run_table(tmp_path, "gpt5mini", "baseline-r2")
        assert list(table["run_type"]) == ["baseline-r2"] * 3
        ont = table[table["category"] == "Ontology-constrained"].iloc[0]
        non = table[table["category"] == "Non-ontology-constrained"].iloc[0]
        assert ont["accuracy"].startswith("1.00")  # tissue right in both records
        assert non["accuracy"].startswith("0.00")  # title wrong in both
        assert ont["n_records"] == 2

    def test_table_has_an_interval_for_every_metric(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        table = build_single_run_table(tmp_path, "gpt5mini", "agent-tool")
        for column in ("accuracy", "precision", "recall", "f1"):
            for value in table[column]:
                assert "[" in value and "]" in value, (column, value)

    def test_empty_run_is_reported_rather_than_raising(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        table = build_single_run_table(tmp_path, "gpt5mini", "never-ran")
        assert list(table["n_records"]) == [0, 0, 0]


class TestAdjustPvalues:
    def test_holm_is_the_step_down_bonferroni(self) -> None:
        # n=4: 0.01*4=0.04, 0.02*3=0.06, 0.03*2=0.06 (held up by the previous), 0.04*1=0.06.
        assert adjust_pvalues([0.01, 0.02, 0.03, 0.04], "holm") == pytest.approx([0.04, 0.06, 0.06, 0.06])

    def test_benjamini_hochberg_is_less_conservative(self) -> None:
        holm = adjust_pvalues([0.01, 0.02, 0.03, 0.04], "holm")
        bh = adjust_pvalues([0.01, 0.02, 0.03, 0.04], "fdr_bh")
        assert all(b <= h for b, h in zip(bh, holm, strict=True))
        assert bh == pytest.approx([0.04, 0.04, 0.04, 0.04])

    def test_order_is_preserved(self) -> None:
        adjusted = adjust_pvalues([0.9, 0.001, 0.02], "holm")
        assert adjusted[1] < adjusted[2] < adjusted[0], "the ranking must survive the correction"

    def test_ties_at_the_cap_stay_monotone(self) -> None:
        """Large p-values all clamp to 1.0, which is correct rather than an ordering bug."""
        adjusted = adjust_pvalues([0.9, 0.001, 0.5], "holm")
        assert adjusted == pytest.approx([1.0, 0.003, 1.0])

    def test_never_exceeds_one_and_handles_empty(self) -> None:
        assert all(p <= 1.0 for p in adjust_pvalues([0.5, 0.6, 0.9], "holm"))
        assert adjust_pvalues([], "holm") == []

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="holm"):
            adjust_pvalues([0.1], "bonferroni")


class TestPairedPermutationPrf:
    def test_no_difference_gives_p_of_one(self) -> None:
        confusion = [(5, 2, 3, 5, 2, 3)] * 8  # the two arms are identical
        result = paired_permutation_prf(confusion)
        assert result["n_effective"] == 0
        assert result["precision"]["pvalue"] == 1.0
        assert result["precision"]["delta"] == 0.0

    def test_consistent_advantage_is_detected(self) -> None:
        confusion = [(4, 6, 6, 9, 1, 1)] * 10  # ARMS better in every record
        result = paired_permutation_prf(confusion)
        assert result["precision"]["delta"] > 0
        assert result["precision"]["pvalue"] < 0.01
        assert result["n_effective"] == 10

    def test_delta_matches_the_bootstrap_point_estimate(self) -> None:
        confusion = [(4, 6, 6, 9, 1, 1), (5, 5, 4, 7, 3, 2), (3, 7, 8, 6, 4, 5)]
        test = paired_permutation_prf(confusion)
        ci = cluster_bootstrap_prf_delta(confusion)
        for metric in ("precision", "recall", "f1"):
            assert test[metric]["delta"] == pytest.approx(ci[metric][0])

    def test_cannot_reach_significance_with_too_few_records(self) -> None:
        """2**n arrangements bound the smallest p, however large the effect."""
        for n_records in (3, 4, 5):
            confusion = [(0, 10, 10, 10, 0, 0)] * n_records  # maximally extreme
            assert paired_permutation_prf(confusion)["precision"]["pvalue"] > 0.05, n_records
        confusion = [(0, 10, 10, 10, 0, 0)] * 6
        assert paired_permutation_prf(confusion)["precision"]["pvalue"] < 0.05


class TestBuildPerAssayPrecisionRecallTable:
    def test_rows_columns_and_correction(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        table = build_per_assay_precision_recall_table(tmp_path, "gpt5mini")
        assert set(table["metric"]) == {"precision", "recall"}, "F1 must stay out of the family"
        for column in ("baseline", "agent-tool", "difference", "perm_p", "p_adjusted", "significant"):
            assert column in table.columns
        # One assay x 3 categories x 2 metrics, minus categories absent from the data.
        assert len(table) == len(set(zip(table["category"], table["metric"], strict=True)))

    def test_correction_makes_it_harder_to_call_significant(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        holm = build_per_assay_precision_recall_table(tmp_path, "gpt5mini", correction="holm")
        bh = build_per_assay_precision_recall_table(tmp_path, "gpt5mini", correction="fdr_bh")
        assert list(holm["perm_p"]) == list(bh["perm_p"]), "the raw test must not depend on the correction"
        assert holm["significant"].sum() <= bh["significant"].sum()

    def test_empty_root_returns_empty_frame(self, tmp_path: Path) -> None:
        assert build_per_assay_precision_recall_table(tmp_path, "gpt5mini").empty


class TestConditionAgnosticComparison:
    """Any two conditions can be paired, not only baseline against ARMS."""

    @staticmethod
    def _add_condition(root: Path, run_type: str, record: dict) -> None:
        for name in ("r1.json", "r2.json"):
            _write_record(root / "atacseq" / "output" / "gpt5mini" / run_type / name, record)

    def test_any_two_conditions_can_be_paired(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        self._add_condition(tmp_path, "baseline-r2", {"tissue": "lung", "title": "WRONG"})
        self._add_condition(tmp_path, "baseline-r3", {"tissue": "lung", "title": "study"})

        data = collect_paired_data(
            tmp_path, "gpt5mini", "atacseq", baseline_run="baseline-r2", system_run="baseline-r3"
        )
        assert len(data.record_acc["all"]) == 2
        # baseline-r2 gets the title wrong, baseline-r3 gets it right.
        assert data.field_outcomes["non_ontology"][0] == (False, True)

    def test_swapping_the_runs_swaps_the_arms(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        forward = collect_paired_data(tmp_path, "gpt5mini", "atacseq")
        reverse = collect_paired_data(tmp_path, "gpt5mini", "atacseq", baseline_run="agent-tool", system_run="baseline")
        assert forward.field_outcomes["ontology"][0] == (False, True)
        assert reverse.field_outcomes["ontology"][0] == (True, False)

    def test_swapping_the_runs_flips_the_sign_of_the_difference(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        forward = collect_paired_data(tmp_path, "gpt5mini", "atacseq")
        reverse = collect_paired_data(tmp_path, "gpt5mini", "atacseq", baseline_run="agent-tool", system_run="baseline")
        ahead = paired_permutation_prf(forward.record_confusion["all"])
        behind = paired_permutation_prf(reverse.record_confusion["all"])
        for metric in ("precision", "recall", "f1"):
            assert ahead[metric]["delta"] == pytest.approx(-behind[metric]["delta"])
            assert ahead[metric]["pvalue"] == behind[metric]["pvalue"], "the test is two-sided"

    def test_columns_are_named_after_the_runs(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        self._add_condition(tmp_path, "baseline-r2", {"tissue": "lung", "title": "WRONG"})
        table = build_precision_recall_table(tmp_path, "gpt5mini", baseline_run="baseline-r2", system_run="agent-tool")
        assert "baseline-r2" in table.columns
        assert "agent-tool" in table.columns
        assert "baseline" not in table.columns, "the column must follow the run that was asked for"

    def test_default_still_compares_baseline_against_arms(self, tmp_path: Path) -> None:
        _build_mini_data_root(tmp_path)
        explicit = collect_paired_data(
            tmp_path, "gpt5mini", "atacseq", baseline_run="baseline", system_run="agent-tool"
        )
        assert collect_paired_data(tmp_path, "gpt5mini", "atacseq").record_acc == explicit.record_acc


class TestRunSelector:
    def test_unknown_selector_raises_rather_than_silently_picking_the_other_run(self) -> None:
        """A run type passed where a role belongs must fail loudly, not select the wrong run."""
        counts = [(1, 2, 3), (2, 2, 4)]
        for bad in ("agent-tool", "arms-agent", "arms", ""):
            with pytest.raises(ValueError, match="baseline"):
                cluster_bootstrap_pooled(counts, bad)
            with pytest.raises(ValueError, match="baseline"):
                cluster_bootstrap_prf([(1, 2, 3, 4, 5, 6)], bad)

    def test_the_two_roles_are_accepted(self) -> None:
        # (baseline_correct, system_correct, total) per record, pooled over both records.
        counts = [(1, 2, 3), (2, 2, 4)]
        assert cluster_bootstrap_pooled(counts, "baseline")[0] == 3 / 7
        assert cluster_bootstrap_pooled(counts, "system")[0] == 4 / 7
