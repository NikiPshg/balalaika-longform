"""A22-voxcpm (owner rule 2026-09-05, RAM budget): streaming safetensors writer.

`safetensors.torch.save_file` materialises the bytes of EVERY tensor on the CPU before
writing (`_flatten` -> `_tobytes` for all keys), i.e. a full ~9.2 GB host-RAM duplicate
of the 2B fp32 state dict during each checkpoint -- the burst that, together with a
second concurrent training, exhausted the 62 GB host on 2026-09-02.

This writer produces a valid safetensors file (8-byte little-endian header length, JSON
header padded with spaces to a multiple of 8, tensors laid out in dict order with
`data_offsets`; the reference writer sorts tensors differently in the data region, the
per-tensor bytes are identical), but moves 64-MB chunks of ONE tensor at a time to the
CPU and writes them straight to the file, so host peak is ~64 MB instead of the whole
state dict. Output loads with `safetensors.torch.load_file` / `safe_open` unchanged
(verified against save_file in the self-test at the bottom).
"""
from __future__ import annotations

import json
import os

import torch

_DTYPE_NAMES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8",
    torch.uint8: "U8", torch.bool: "BOOL",
}


def save_file_streaming(tensors: dict, path: str, metadata: dict | None = None,
                        chunk_bytes: int = 64 << 20) -> int:
    """Write `tensors` (name -> torch.Tensor, any device) as a safetensors file.

    Returns the number of bytes written. Keys are written in dict order (like
    save_file, which sorts nothing but the header keys inside JSON; we keep dict
    order for the data region and let JSON keep insertion order).
    """
    header = {}
    offset = 0
    for name, t in tensors.items():
        if t.layout != torch.strided:
            raise ValueError(f"sparse tensor not supported: {name}")
        nbytes = t.numel() * t.element_size()
        header[name] = {
            "dtype": _DTYPE_NAMES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata:
        header = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}, **header}
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(hjson) % 8) % 8
    hjson += b" " * pad

    tmp = path + ".tmp"
    written = 0
    with open(tmp, "wb") as f:
        f.write(len(hjson).to_bytes(8, "little"))
        f.write(hjson)
        written += 8 + len(hjson)
        for name, t in tensors.items():
            cpu = t.detach()
            if not cpu.is_contiguous():
                cpu = cpu.contiguous()
            # move at most `chunk_bytes` of one tensor to the host at a time
            flat = cpu.view(-1)
            elt = flat.element_size()
            step = max(1, chunk_bytes // elt)
            for i in range(0, flat.numel(), step):
                piece = flat[i:i + step].to("cpu", copy=False)
                if piece.dtype == torch.bfloat16:
                    piece = piece.view(torch.int16)   # numpy has no bf16; raw bytes are identical
                buf = piece.contiguous().numpy().tobytes()
                f.write(buf)
                written += len(buf)
                del piece, buf
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return written


if __name__ == "__main__":  # self-test against the reference writer
    import tempfile

    from safetensors.torch import load_file, save_file

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = {
        "a.weight": torch.randn(73448, 64, device=dev),
        "b.bias": torch.randn(2048, device=dev).to(torch.bfloat16),
        "c.scalar": torch.tensor(3.5, device=dev),
        "d.int": torch.arange(10, device=dev, dtype=torch.int32),
    }
    with tempfile.TemporaryDirectory() as d:
        ref = os.path.join(d, "ref.safetensors")
        out = os.path.join(d, "stream.safetensors")
        save_file({k: v.cpu().contiguous() for k, v in sd.items()}, ref)
        save_file_streaming(sd, out)
        r, s = load_file(ref), load_file(out)
        assert r.keys() == s.keys()
        for k in r:
            assert r[k].dtype == s[k].dtype and r[k].shape == s[k].shape, k
            assert torch.equal(r[k], s[k]), k
        # byte-level: every tensor's raw bytes (located via each file's own header offsets)
        # must be identical; the reference writer orders tensors differently in the data
        # region (sorted), so whole-region equality is not expected.
        def _raw(path):
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                hdr = json.loads(f.read(n))
                base = 8 + n
                out = {}
                for k, v in hdr.items():
                    if k == "__metadata__":
                        continue
                    b, e = v["data_offsets"]
                    f.seek(base + b)
                    out[k] = f.read(e - b)
                return out
        r_raw, s_raw = _raw(ref), _raw(out)
        assert r_raw.keys() == s_raw.keys()
        for k in r_raw:
            assert r_raw[k] == s_raw[k], f"raw bytes differ for {k}"
        print("self-test OK: streaming writer == save_file (tensors equal, per-tensor bytes identical)")
