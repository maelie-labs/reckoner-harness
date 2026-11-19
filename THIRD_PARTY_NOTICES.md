# Third-Party Software Notices and Licenses

This document contains licensing notices and copyright information for third-party software components bundled into or statically linked with the Reckoner Harness (`reckoner-harness`).

All components are distributed under permissive, Apache-2.0 compatible open-source licenses.

---

## 1. Lua 5.4

- **Component**: Lua 5.4.9 (official PUC-Rio C source)
- **License**: MIT License
- **Copyright**: (c) 1994–2026 Lua.org, PUC-Rio.
- **URL**: https://www.lua.org/

```
MIT License

Copyright (c) 1994–2026 Lua.org, PUC-Rio.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. The Go Standard Library & Runtime

- **Component**: Go Runtime and Standard Library
- **License**: BSD-3-Clause License
- **Copyright**: (c) 2009 The Go Authors. All rights reserved.
- **URL**: https://go.dev/

```
Copyright (c) 2009 The Go Authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
   * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 3. musl libc (Linux Targets)

- **Component**: musl libc (statically linked on Linux x86_64 and aarch64)
- **License**: MIT License
- **Copyright**: (c) 2005–2020 Rich Felker, et al.
- **URL**: https://musl.libc.org/

```
musl as a whole is licensed under the following standard MIT license:

Copyright © 2005-2020 Rich Felker, et al.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

---

## 4. mingw-w64 CRT Startup (Windows Targets)

- **Component**: mingw-w64 runtime startup code (linked into Windows PE binaries)
- **License**: Zope Public License (ZPL) Version 2.1
- **Copyright**: Copyright (c) 2009 - 2013 by the mingw-w64 project
- **URL**: https://www.mingw-w64.org/

```
With exception of certain parts that are prominently marked as being
in the Public Domain, BSD, or LGPL this Software is provided under the
Zope Public License (ZPL) Version 2.1.

Copyright (c) 2009 - 2013 by the mingw-w64 project

See the AUTHORS file for the list of contributors to the mingw-w64 project.

This license has been certified as open source. It has also been designated
as GPL compatible by the Free Software Foundation (FSF).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   1. Redistributions in source code must retain the accompanying copyright
      notice, this list of conditions, and the following disclaimer.
   2. Redistributions in binary form must reproduce the accompanying
      copyright notice, this list of conditions, and the following disclaimer
      in the documentation and/or other materials provided with the
      distribution.
   3. Names of the copyright holders must not be used to endorse or promote
      products derived from this software without prior written permission
      from the copyright holders.
   4. The right to distribute this software or to use it for any purpose does
      not give you the right to use Servicemarks (sm) or Trademarks (tm) of
      the copyright holders.  Use of them is covered by separate agreement
      with the copyright holders.
   5. If any files are modified, you must cause the modified files to carry
      prominent notices stating that you changed the files and the date of
      any change.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESSED
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

*Note on Public Domain Components*: The linked `pseudo-reloc.c` runtime component is in the public domain:
> *"THIS SOFTWARE IS NOT COPYRIGHTED. This source code is offered for use in the public domain. You may use, modify or distribute it freely."*

---

## 5. Zig compiler_rt (Compiler Builtins)

- **Component**: Zig compiler_rt runtime builtins (linked into compiled binaries)
- **License**: The MIT License (Expat)
- **Copyright**: Copyright (c) Zig contributors
- **URL**: https://ziglang.org/

```
The MIT License (Expat)

Copyright (c) Zig contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## 6. jsonschema (Go JSON Schema Validator)

- **Component**: `github.com/santhosh-tekuri/jsonschema/v5`
- **License**: Apache License, Version 2.0
- **Copyright**: Santhosh Kumar Tekuri
- **URL**: https://github.com/santhosh-tekuri/jsonschema

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


---

## 7. golang.org/x/text (Unicode normalization for `find`)

- **Component**: `golang.org/x/text` v0.43.0 (used by the BM25 tokenizer in `bm25.go`)
- **License**: BSD-3-Clause License
- **Copyright**: Copyright 2009 The Go Authors.
- **URL**: https://pkg.go.dev/golang.org/x/text

Same BSD-3-Clause license text as the Go standard library in section 2.

---

## 8. Embedded Scientific Reference Data (`lookup_data/`)

The `lookup` tool embeds three scientific reference tables compiled into the binary (`lookup_data/constants.json`, `lookup_data/elements.json`, `lookup_data/astronomy.json`; see `lookup_data/SOURCES.md` for full provenance and field definitions):

1. **Fundamental Physical Constants (`lookup_data/constants.json`)**:
   - **Source**: NIST CODATA 2022 Fundamental Physical Constants (`allascii.txt`), National Institute of Standards and Technology, Physical Measurement Laboratory (U.S. Department of Commerce).
   - **License / Status**: NIST Standard Reference Database 121 (`15 U.S.C. § 290e`). Under the Standard Reference Data Act (`15 U.S.C. § 290e`), the U.S. Secretary of Commerce may copyright SRD compilations; NIST PML provides the CODATA fundamental physical constants table (`allascii.txt`) openly to the public worldwide for unrestricted scientific, commercial, and educational use without licensing fees or usage restrictions (see `lookup_data/SOURCES.md`).
   - **URL**: https://physics.nist.gov/cuu/Constants/
2. **Periodic Table of Elements (`lookup_data/elements.json`)**:
   - **Source**: NCBI PubChem Periodic Table of Elements (`PubChemElements_all.json`), National Library of Medicine / National Institutes of Health (U.S. Department of Health and Human Services), with IUPAC standard atomic weights.
   - **License / Status**: Public Domain in the United States pursuant to `17 U.S.C. § 105`.
   - **URL**: https://pubchem.ncbi.nlm.nih.gov/periodic-table/
3. **Solar System & IAU Nominal Conversion Constants (`lookup_data/astronomy.json`)**:
   - **Source**: NASA Goddard Space Flight Center Planetary Fact Sheet (`https://nssdc.gsfc.nasa.gov/planetary/factsheet/`) and International Astronomical Union (IAU) 2015 Resolution B3 on Recommended Nominal Conversion Constants for Selected Solar and Planetary Properties (`arXiv:1510.07674`).
   - **License / Status**: Public Domain in the United States pursuant to `17 U.S.C. § 105` (NASA GSFC) and public scientific resolution constants (IAU 2015 Resolution B3).

---

## 9. Embedded Tokenizer (`tokenizer_data/`)

- **Component**: Reckoner Tokenizer v2 (`tokenizer_data/tokenizer.json.gz`, 36,864 tokens; see `tokenizer_data/SOURCES.md`)
- **License**: Apache License, Version 2.0
- **Copyright**: Copyright 2026 Maelie Labs.

---

## 10. SGLang Integration Plugin (`integrations/sglang/`) & External Python Runtime Dependencies

The optional Python SGLang integration plugin in `integrations/sglang/` is distributed under the Apache License, Version 2.0 (Copyright 2026 Maelie Labs) and includes adaptations for the following permissive open-source upstream projects:

1. **SGLang (`sglang` v0.5.22 and `integrations/sglang/patches/0001-inflight-extend.patch`)**:
   - **License**: Apache License, Version 2.0
   - **Copyright**: Copyright 2023-2024 SGLang Team; modifications Copyright 2026 Maelie Labs.
   - **URL**: https://github.com/sgl-project/sglang
2. **Qwen3.5 Architecture Adaptation (`integrations/sglang/sglang_reckoner/modeling_qwen35_gdn2.py` and `config.py`)**:
   - **License**: Apache License, Version 2.0
   - **Copyright**: Copyright 2025 The Qwen Team, Alibaba Group, and the HuggingFace Inc. team; GDN-2 modifications Copyright 2026 Maelie Labs.
   - **URL**: https://huggingface.co/Qwen/Qwen3.5-2B

### External Python Runtime Dependencies (Not Bundled)

When installing the optional Python serving environment (`integrations/sglang/pyproject.toml` or `integrations/sglang/sglang_reckoner/sglang_rl_env_lock.txt`), pip/uv fetches third-party Python packages from PyPI that are not bundled in this repository or its compiled binaries:

- **Direct plugin runtime dependencies (`pyproject.toml`)**:
  - `sglang==0.5.22`, `transformers>=4.48.0`, `tokenizers>=0.21.0`, `compressed-tensors>=0.9.0`, `grpcio>=1.60.0`: Apache-2.0
  - `torch>=2.6.0`, `numpy>=1.26.0`, `protobuf>=4.25.0`: BSD-3-Clause
  - `flash-linear-attention==0.6.0` (with `fla-core==0.6.0`): MIT License
- **Pinned reference environment lockfile (`sglang_rl_env_lock.txt`)**:
  - Permissive open-source packages (`Apache-2.0`, `MIT`, `BSD-3-Clause`, `BSD-2-Clause`, `ISC`, `PSF-2.0`, `MPL-2.0`), including `triton==3.8.0` (MIT License, installed as a dependency of `torch==2.14.1+cu130`).
  - NVIDIA CUDA 13 runtime wheels (`nvidia-cublas`, `nvidia-cuda-cccl`, `nvidia-cuda-crt`, `nvidia-cuda-cupti`, `nvidia-cuda-nvcc`, `nvidia-cuda-nvrtc`, `nvidia-cuda-runtime`, `nvidia-cudnn-cu13`, `nvidia-cudnn-frontend`, `nvidia-cufft`, `nvidia-cufile`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-cusparse`, `nvidia-cusparselt-cu13`, `nvidia-cutlass-dsl`, `nvidia-cutlass-dsl-libs-base`, `nvidia-cutlass-dsl-libs-core`, `nvidia-cutlass-dsl-libs-cu12`, `nvidia-cutlass-dsl-libs-cu13`, `nvidia-mathdx`, `nvidia-ml-py`, `nvidia-nccl-cu13`, `nvidia-nvjitlink`, `nvidia-nvshmem-cu13`, `nvidia-nvtx`, `nvidia-nvvm`, `cuda-bindings`, `cuda-core`, `cuda-pathfinder`, `cuda-python`, `cuda-tile`, `cuda-toolkit`): governed by the NVIDIA Software License Agreement / CUDA Supplement to Software License Agreement when installed by the user on CUDA hosts.
  - Transitive dependencies of upstream `sglang` / `outlines` present in the full lockfile (`easydict==1.13` under LGPL-3.0, `pycountry==26.2.16` under LGPL-2.1-only): neither package is imported or distributed by `sglang_reckoner` or `reckoner-harness`.
