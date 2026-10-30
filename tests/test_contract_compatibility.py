"""The release contract gate detects drift without a checked-in schema baseline."""

from __future__ import annotations

from scripts.check_contract_compatibility import ContractSnapshot, _compare


def _alpha_snapshot() -> ContractSnapshot:
    return ContractSnapshot(
        package_file="alpha/src/attitude_convention_inference/__init__.py",
        root_exports=("infer_conventions", "InferenceReport"),
        model_exports=("InferenceReport", "RateFrame"),
        function_signatures={"infer_conventions": "(data, config=None)"},
        class_signatures={"InferenceReport": "(best)"},
        enum_members={"RateFrame": (("BODY", "'body'"), ("WORLD", "'world'"))},
        model_schema_sha256={"InferenceReport": "alpha-schema"},
    )
####


def test_contract_gate_allows_additive_exports() -> None:
    alpha = _alpha_snapshot()
    current = alpha.model_copy(
        update={
            "root_exports": (*alpha.root_exports, "new_helper"),
            "model_exports": (*alpha.model_exports, "NewReport"),
            "model_schema_sha256": {**alpha.model_schema_sha256, "NewReport": "new-schema"},
        }
    )

    assert _compare(alpha, current) == []
####


def test_contract_gate_reports_removed_or_changed_alpha_contracts() -> None:
    alpha = _alpha_snapshot()
    changed = alpha.model_copy(
        update={
            "root_exports": ("InferenceReport",),
            "model_exports": ("RateFrame",),
            "function_signatures": {"infer_conventions": "(data)"},
            "class_signatures": {"InferenceReport": "(best, extra)"},
            "enum_members": {"RateFrame": (("BODY", "'body'"),)},
            "model_schema_sha256": {"InferenceReport": "changed-schema"},
        }
    )

    assert _compare(alpha, changed) == [
        "package exports removed: infer_conventions",
        "model exports removed: InferenceReport",
        "function signature changed: infer_conventions",
        "class signature changed: InferenceReport",
        "enum members changed: RateFrame",
        "model JSON schema changed: InferenceReport",
    ]
####
