# Tokenizer Sources and Provenance

- **Canonical Model Tokenizer**: Reckoner Tokenizer v2 (`tokenizer.json.gz`, vocab size 36,864).
- **Training Method**: Byte-Level Byte-Pair Encoding (BPE) trained with individual digit splitting via `pre_tokenizers.Digits(individual_digits=True)`.
- **Training Corpus**: Curated pretraining corpus comprising Cosmopedia-v2 synthetic textbooks and expositions, FineWeb-Edu educational web text, Open-Web-Math, and Stack-Edu high-quality code.
- **Vocabulary Layout & Architecture Alignment**:
  - Total vocabulary size: 36,864 tokens (288 × 128, aligned to 128-wide Tensor Core tiles).
  - Control and special tokens: IDs 0..127:
    - Sequence & turn control: IDs 0..14 (`<|endoftext|>`, `<|pad|>`, `<|turn:user|>`, `<|turn:agent|>`, `<|call|>`, `<|/call|>`, etc.)
    - Observation delimiters: IDs 15..17 (`<|obs|>`, `<|obs:err|>`, `<|/obs|>`)
    - Tool call delimiters: IDs 18..20, 29..32, 35..36 (`<|call:calc|>`, `<|call:solve|>`, `<|call:lua|>`, `<|call:str|>`, `<|call:date|>`, `<|call:lookup|>`, `<|call:find|>`, etc.)
    - Circular registers: IDs 21..28 and 37..44 (`<|r0|>`..`<|r15|>`)
    - Context quote delimiters: IDs 33..34 (`<|quote|>`, `<|/quote|>`)
    - Reserved special tokens: IDs 45..127 (`<|reserved_16|>`..`<|reserved_98|>`)
  - Base byte tokens: IDs 128..255 (individual byte values mapped via Byte-Level BPE unicode characters).
  - Ordinary text BPE tokens: IDs 256..32,895 (BPE merges; IDs 32,768..32,895 are ordinary vocabulary words such as `Environ` at ID 32,768).
  - Context message tokens: IDs 32,896..33,151 (`<|m0|>`..`<|m255|>`, 256 message slots).
  - Context line anchor tokens: IDs 33,152..34,175 (`<|L0|>`..`<|L1023|>`, 1,024 line anchors).
  - Reserved extension tokens: IDs 34,176..36,863 (`<|reserved_ext_0|>`..`<|reserved_ext_2687|>`).
- **License**: Apache-2.0 (Maelie Labs / Project Reckoner).

