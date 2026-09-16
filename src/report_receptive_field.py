#!/usr/bin/env python3
"""Report the exact pre-adaptive-pooling receptive field of each MST-CNN stream."""

import argparse
import csv
import json
from pathlib import Path


def receptive_field(kernel):
    receptive, jump = 1, 1
    layers = []
    for name, stride in (("block1_conv1", 1), ("block1_conv2", 1), ("block1_stride2_conv", 2), ("block2_conv1", 1), ("block2_conv2", 1)):
        receptive += (kernel - 1) * jump
        jump *= stride
        layers.append({"layer": name, "receptive_field_samples": receptive, "output_stride_samples": jump})
    return receptive, layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", nargs="+", type=int, default=[7, 9, 11, 13])
    parser.add_argument("--sfreq", type=float, default=160.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, details = [], {}
    for kernel in args.kernels:
        samples, layers = receptive_field(kernel)
        rows.append({
            "kernel_samples": kernel,
            "kernel_duration_ms": 1000.0 * kernel / args.sfreq,
            "effective_receptive_field_samples": samples,
            "effective_receptive_field_ms": 1000.0 * samples / args.sfreq,
            "closed_form": f"7*{kernel}-6",
        })
        details[str(kernel)] = layers
    with (output / "receptive_field.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "sampling_frequency_hz": args.sfreq,
        "architecture": "two ConvBlocks: Conv, Conv, stride-2 Conv, Conv, Conv",
        "pre_adaptive_pooling_receptive_field": rows,
        "layerwise_calculation": details,
        "interpretation": "The convolutional pathway spans 7k-6 samples before adaptive pooling. Adaptive pooling and the dense classifier then integrate features across the full selected epoch.",
    }
    (output / "receptive_field.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
