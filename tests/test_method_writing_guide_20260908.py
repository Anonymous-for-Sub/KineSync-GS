import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUIDE = PROJECT_ROOT / "paper-handoff-20260908" / "METHOD_AND_WRITING_GUIDE_ZH.md"
TABLES = PROJECT_ROOT / "paper-handoff-20260908" / "KINESYNC_GS_PAPER_TABLES_20260908.tex"
EVIDENCE = PROJECT_ROOT / "code" / "src" / "kinesync" / "guard" / "evidence.py"
REAL_ROUTE_A = PROJECT_ROOT / "code" / "src" / "kinesync" / "observation" / "real_route_a.py"


class MethodWritingGuideTest(unittest.TestCase):
    def test_guide_preserves_evidence_scope_and_paper_tables(self):
        guide = GUIDE.read_text(encoding="utf-8")
        tables = TABLES.read_text(encoding="utf-8")
        for phrase in (
            "q^+ = q^m + S",
            r"\Delta_{\mathrm{vis}}",
            r"\max(|\mathcal L(q^m)|,\epsilon)",
            r"P_I\nabla_q",
            r"\max(\|\delta_{\mathrm{head}}\|+\|\delta_{\mathrm{extra}}\|,\epsilon)",
            "mean_mask_iou",
            r"\Sigma_{k}(q)=R_{\ell(k)}(q)\Sigma_{k,0}R_{\ell(k)}(q)^{\mathsf T}",
            "retrospective shadow replay",
            "不等同于机器人任务成功",
            "37.96% < 40.00%",
            "component-level qMAE",
            "OpenVLA-OFT policy interface",
            "22,562",
            "0.05834",
            "Q4__paper-policy-interface-actions",
            "协议名称与策略完整性",
            "外部 risk-atomic",
            "独立外部留出扩展",
            "0.322",
            "12/13",
            "0.00097656",
            "Holm",
            "gain $\\geq 0$",
            "0.4267100393772125",
            "非限制性检查",
            "\\tau_{\\mathrm{RT3}}",
            "\\tau_{\\mathrm{RT6}}",
        ):
            self.assertIn(phrase, guide)
        self.assertIn("0.00247", tables)
        self.assertIn("0.08406", tables)
        self.assertIn("Spatial synchronization full matched matrix", tables)
        self.assertIn("Component-level residual diagnostic", tables)
        self.assertNotIn("Component qMAE \\\\n", tables)
        rt3_start = guide.index("### 1.2")
        atomic_start = guide.index("### 2.2")
        historical_guard = guide[rt3_start:atomic_start]
        self.assertNotIn("0.4267100393772125", historical_guard)
        self.assertNotIn("gain $\\geq 0$", historical_guard)
        self.assertIn("caption", tables)

    def test_guide_math_tracks_implementation_tokens(self):
        evidence = EVIDENCE.read_text(encoding="utf-8")
        route_a = REAL_ROUTE_A.read_text(encoding="utf-8")
        self.assertIn("before_loss.abs().clamp_min(epsilon)", evidence)
        self.assertIn("denominator = first_norm + second_norm", evidence)
        self.assertIn("return gradient[indices].detach().clone()", evidence)
        for token in ("mean_mask_iou", "mean_boundary", "mean_rgb", "weights.prior * prior"):
            self.assertIn(token, route_a)


if __name__ == "__main__":
    unittest.main()
