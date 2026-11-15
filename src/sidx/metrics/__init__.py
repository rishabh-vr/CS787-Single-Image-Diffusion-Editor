"""Evaluation metrics (plan §8, groups G1-G5; definitions in docs/technical_notes.md §5.3).

    paper       the paper's Table-1 metrics: SIFID, pixel/LPIPS diversity, NIQE/NIMA/MUSIQ,
                near-duplicate rate (Table S1)
    patchdist   domain-agnostic distribution fidelity: patch SWD, patch precision/recall
    copying     verbatim copying: large-block copy ratio from the nearest-patch offset field
    structure   structural validity for line art / patterns: line statistics (LSD), skeleton
                endpoints and junctions, grey-level ghosting, edges, periodicity

Images are (C, H, W) float tensors in [0, 1] unless stated otherwise. Everything runs on CPU;
networks (Inception, AlexNet, IQA models) are loaded once and cached.
"""
