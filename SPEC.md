# Reckoner Harness Protocol Specification

**Specification Version**: 1.1.3  
**Target Runtimes**: Reckoner Model Family, Standalone and Agent Runtimes  
**Status**: Public Specification (v1.1.3)  

## Normative Language and Conformance Requirements

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in BCP 14, RFC 2119. A conforming implementation of the Reckoner Harness MUST satisfy all MUST, MUST NOT, REQUIRED, SHALL, and SHALL NOT requirements.

---

## 1. Overview and Architecture

The Reckoner Harness is an out-of-core deterministic execution engine, context reference store, and variable container for language model inference. Instead of writing verbose JSON tool calls or running unconstrained code in a slow interpreter, the model emits compact, dedicated control tokens. The harness intercepts these tokens at inference time, executes operations in microsecond-scale native Go code, stores typed values in circular registers, manages byte-exact context references, and returns canonical observation strings into the model context.

### 1.1 Division of Responsibilities

- **Language Model**: Acts as the semantic reasoning and planning controller. It decides when to invoke a tool, formulates mathematical or algorithmic expressions, resolves citations, and consumes observations to produce downstream reasoning or final answers.
- **Harness Engine**: Acts as the deterministic execution engine, context store, type checker, and state container. It provides exact precision, enforces resource and execution step limits, maintains register state across turns, and manages line-level context indexing and quotes.

```
+-------------------------------------------------------+
|                 Language Model Core                   |
|  Generates: <|call:calc|> 482 * 19 - 14 <|/call|>     |
+-------------------------------------------------------+
                           |
                           v (Token Interception)
+-------------------------------------------------------+
|               Reckoner Native Harness                 |
|  - Parse infix expression                             |
|  - Execute arbitrary-precision integer arithmetic     |
|  - Auto-bind result into register <|r0|>              |
+-------------------------------------------------------+
                           |
                           v (Observation Injection)
+-------------------------------------------------------+
|                 Language Model Context                |
|  Receives: <|obs|><|r0|> = 9144<|/obs|>               |
+-------------------------------------------------------+
```

---

## 2. Control Tokens and Delimiter Grammar

The protocol relies on dedicated atomic delimiters for tool dispatch, observation ingestion, register reference, and context citation.

### 2.1 Delimiter Table

| Delimiter | Role | Description |
| :--- | :--- | :--- |
| `<\|call:calc\|>` | Call Open | Invokes the exact arithmetic, polynomial, geometric, and linear algebra engine. |
| `<\|call:solve\|>` | Call Open | Invokes the symbolic equation and inequality solver. |
| `<\|call:lua\|>` | Call Open | Invokes the embedded Lua 5.4 algorithmic sandbox. |
| `<\|call:str\|>` | Call Open | Invokes the multibyte string manipulation engine. |
| `<\|call:date\|>` | Call Open | Invokes the civil calendar date arithmetic engine. |
| `<\|call:lookup\|>` | Call Open | Invokes the offline scientific and physical constants reference table. |
| `<\|call:find\|>` | Call Open | Invokes the long-context literal substring search and line range extraction engine (`find` and `lines`). |
| `<\|call\|>` | Call Open | Invokes an external agent function (see Section 11). |
| `<\|/call\|>` | Call Close | Closes an active tool call payload. Triggers immediate harness evaluation. |
| `<\|obs\|>` | Observation Open | Opens a successful observation returned by the harness. |
| `<\|obs:err\|>` | Observation Error Open | Opens an error observation returned by the harness. |
| `<\|/obs\|>` | Observation Close | Closes an observation sequence. |
| `<\|quote\|>` | Quote Open | Opens an anchored context citation reference. |
| `<\|/quote\|>` | Quote Close | Closes an anchored context citation reference. |
| `<\|m0\|>`..`<\|m255\|>` | Message Anchor | Delimits message boundaries in the active context store (256 cyclical slots). |
| `<\|L0\|>`..`<\|L1023\|>` | Line Anchor | Delimits line and mid-line anchor points within a message (1,024 slots per message). |
| `<\|r0\|>`..`<\|r15\|>` | Register Reference | Identifies one of the 16 circular working memory registers. |

### 2.2 Delimiter Grammar

A conforming model turn containing tool invocations MUST adhere to the following EBNF grammar:

```ebnf
Turn               ::= ( Text | ToolInvocation | ContextQuote )*
ToolInvocation     ::= CallOpen ToolPayload CallClose Observation
CallOpen           ::= "<|call:calc|>" | "<|call:solve|>" | "<|call:lua|>"
                     | "<|call:str|>" | "<|call:date|>" | "<|call:lookup|>"
                     | "<|call:find|>" | "<|call|>"
CallClose          ::= "<|/call|>"
ToolPayload        ::= [^<]+
Observation        ::= ( "<|obs|>" ObsContent "<|/obs|>" )
                     | ( "<|obs:err|>" ErrorContent "<|/obs|>" )
ContextQuote       ::= "<|quote|>" MessageAnchor LineAnchor LineAnchor "<|/quote|>"
MessageAnchor      ::= "<|m0|>" | ... | "<|m255|>"
LineAnchor         ::= "<|L0|>" | ... | "<|L1023|>"
RegisterRef        ::= "<|r0|>" | ... | "<|r15|>"
```

### 2.3 Untrusted Input Escaping

Tool observations containing user-supplied or external strings MUST escape delimiter control sequences before injection into the context. Specifically, any literal occurrence of `<|obs|>`, `<|obs:err|>`, `<|/obs|>`, `<|call:`, `<|call|>`, `<|/call|>`, `<|quote|>`, or `<|/quote|>` within payload text MUST be replaced with its escaped literal string to prevent context injection attacks. Across both 1.4B / no-profile execution (`reckoner-1.4b`) and external-tokenizer profiles (`reckoner-q2b`), no tool observation body and no user or client text can inject a non-template control token ID into the token stream. Every in-path tool observation `<|obs|>...<|/obs|>` or `<|obs:err|>...<|/obs|>` in `reckoner-harness serve` and `sglang_reckoner.tool_processor.ReckonerToolLogitProcessor._execute_tool_call` is encoded through a single choke point (`EncodeToolObservation` in Go and `_encode_tool_observation` in Python): the harness emits `<|obs|>` / `<|obs:err|>`, the leading `<|rN|> = ` register prefix on built-in tools (`calc`, `solve`, `str`, `date`, `lookup`, `lua`, `reg`, plus `(<|rN|>)` / `(<|rN|>, replaced ...)` annotations in `solve`), and `<|/obs|>` as its own structure, while replacing `<|` with `<\|` and escaping registered non-`<|...|>` angle-bracket `added_tokens` (`<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` → `<\think>`, `<\/think>`, `<\tool_call>`, `<\/tool_call>`, `<\tool_response>`, `<\/tool_response>`) across the entire tool-produced body (success or error across all built-in and client tools, including `external`/`fncall` observations starting with `<|rN|> = `, `quote` receipts/errors, and `find` and `lines` results quoting the `system` turn) without double-escaping `<\|`. Consequently, `<|...|>` tags written by the harness embedded inside error observation bodies (`ErrUnsetRegister: <|rN|> is empty`, `ErrNotFound: message token <|mN|> not found in active context`, `ErrOutOfBounds: line index L... out of bounds (message <|mN|> has ... lines: ...)`, `ErrSyntax: malformed quote syntax: <|quote|>...<|/quote|>`) and inside `reg list` entries after the first `<|rN|> = ` prefix encode those tags as escaped text (`<\|rN|>`, `<\|mN|>`, `<\|quote|>`, `<\|/quote|>`) on 1.4B / no-profile just as on `reckoner-q2b`; a `find` or `lines` hit on the `<|tools|>...<|/tools|>` line in the `system` turn returns escaped `<\|tools|>` / `<\|/tools|>` text; and pre-formatted `assistant` `content` containing or ending in `<|endoftext|>` or `<|/turn|>` is rejected with `ErrArgs` on 1.4B / no-profile (as on `reckoner-q2b`).

---

## 3. Register Ring Architecture (`<|r0|>` through `<|r15|>`)

The harness maintains 16 circular working memory registers denoted `<|r0|>` through `<|r15|>`.

### 3.1 Auto-Binding Contract

When a tool evaluation finishes successfully without errors:
1. The harness MUST bind the resulting typed value to the next available register in cyclic sequential order (0 → 1 → … → 15 → 0).
2. The harness MUST format the observation as `<|obs|><|rN|> = <result><|/obs|>`.
3. The register binding pointer advances by 1 modulo 16.

### 3.2 Announced Overwrite on Ring Wrap

When more than 16 bindings occur within a session, earlier registers are overwritten in circular order. To ensure full visibility to the attention mechanism, the harness MUST announce overwrites by appending replacement notifications to the observation:

```text
<|obs|><|r0|> = 42 (replaced 3.141593)<|/obs|>
```

### 3.3 Supported Register Value Types

Registers are dynamically typed and preserve exact internal representations across tool evaluations:
- **Integers**: Arbitrary-precision `*big.Int`.
- **Rationals**: Arbitrary-precision exact fractions `*big.Rat`.
- **Surds & Algebraic Numbers**: Closed forms in `ℚ(√d)` and exact radicals.
- **Floats**: IEEE 754 64-bit floating-point numbers.
- **Physical Quantities**: Exact rational magnitudes paired with SI/derived unit dimensions.
- **Strings**: UTF-8 encoded rune sequences.
- **Dates**: Canonical Gregorian civil date records `YYYY-MM-DD (Weekday)`.
- **Intervals**: Open, closed, and disjoint real intervals for inequality solutions.
- **Matrices & Vectors**: Arbitrary-dimension numeric arrays.
- **Polynomials**: Exact rational multivariate polynomials.

### 3.4 Scoping and Lifetime

Register state persists across sequential turns of an inference session. Registers are cleared only when an explicit reset command (`reset`) is issued by the caller.

---

## 4. Exact Number Formatting & Output Specification

To ensure bit-identical cross-platform reproducibility across all tool implementations, a conforming harness MUST format numerical results according to the following strict rules:

### 4.1 Integers
Integers MUST be formatted in standard base-10 decimal notation without exponents, grouping separators, or trailing decimal points (for example, `12345`).

### 4.2 Rationals
1. **Integer Reducibility**: If the denominator is 1, the value MUST format as a plain integer (`r.Num().String()`).
2. **Terminating Decimal Check**: If the denominator factors strictly into powers of 2 and 5 (`q = 2^a · 5^b`) with `max(a, b) ≤ 6`, the rational MUST format as an exact terminating decimal with trailing fractional zeros stripped (for example, `1/8` formats as `0.125`, `1/2` formats as `0.5`, `3/20` formats as `0.15`).
3. **Irreducible Fraction**: If the denominator has prime factors other than 2 and 5, or if `max(a, b) > 6`, the rational MUST format as an irreducible fraction in `p/q` notation (for example, `1/3`, `22/7`, `7/12`).
4. **Forced Fractions**: When wrapped in `frac(expr)`, the value MUST always format in `p/q` notation (or integer if denominator is 1).
5. **Forced Decimals**: When wrapped in `float(expr)` or `approx(expr)`, the value MUST format as a decimal with up to 6 significant digits.
6. **Decimal Rounding**: When evaluated via `round(r, places)`, the rational MUST be rounded using half-away-from-zero rounding (school rounding) to `places` decimal digits, preserving trailing zeros up to `places` decimal places (for example, `round(1/8, 2)` evaluates to `0.13`).

### 4.3 Floating-Point Numbers
1. **Standard Floating-Point**: If `|f| ≥ 10¹⁵` or `|f| < 10⁻⁴` (with `f ≠ 0`), the value MUST format in scientific notation using `%.6g`. Otherwise, standard floats format with `%.6f` with trailing fractional zeros and decimal points stripped.
2. **Exact Integer Float**: If the float represents an exact integer with `|f| ≤ 2⁵³ - 1` (`MaxSafeFloatInt`), it MUST format as `%.0f`.
3. **Invalid Floats**: Floating-point NaN values MUST NOT be bound and MUST emit `ErrDomain`. Floating-point infinite values MUST NOT be bound and MUST emit `ErrOverflow`.

### 4.4 Display Limits
If the formatted representation of any value exceeds 256 characters, the harness MUST format the displayed observation as:
```text
<|obs|><|rN|> = [too large to display: <n> digits]<|/obs|>
```
where `<n>` is the byte length of the formatted value. The exact underlying value MUST remain bound in the register and remain fully usable in subsequent mathematical evaluations.

---

## 5. Embedded Tool Reference Contracts

### 5.1 `calc`: Universal Mathematical AST and Math Engine

The `calc` tool evaluates mathematical expressions via a recursive-descent parser with arbitrary-precision arithmetic.

- **Invocation**: `<|call:calc|> <expression> <|/call|>`
- **Observation**: `<|obs|><|rN|> = <result><|/obs|>`
- **Standard Functions and Operators**:
  - **Arithmetic**: `+`, `-`, `*`, `/`, `//` (floor division), `%` (floor modulus), `^` or `**` (exponentiation), unary `-`.
  - **Elementary Functions**:
    - `sqrt(x)`: Principal square root, returning exact surds when radicand is non-square integer (for example, `sqrt(12)` evaluates to `2*sqrt(3)`).
    - `cbrt(x)`: Principal cube root (for example, `cbrt(27)` evaluates to `3`).
    - `abs(x)`: Absolute value (for example, `abs(-42)` evaluates to `42`).
    - `round(x[, places])`: Half-away-from-zero decimal rounding (for example, `round(1/8, 2)` evaluates to `0.13`).
    - `floor(x)`: Floor function (for example, `floor(7/3)` evaluates to `2`).
    - `ceil(x)`: Ceiling function (for example, `ceil(7/3)` evaluates to `3`).
    - `exp(x)`: Natural exponential function `e^x` (for example, `exp(0)` evaluates to `1`).
    - `ln(x)`: Natural logarithm `ln(x)` (for example, `ln(e)` evaluates to `1`).
    - `log(x[, base])`: Natural logarithm, or logarithm with specified base (for example, `log(8, 2)` evaluates to `3`).
    - `log10(x)`: Common logarithm `log₁₀(x)` (for example, `log10(1000)` evaluates to `3`).
    - `log2(x)`: Binary logarithm `log₂(x)` (for example, `log2(64)` evaluates to `6`).
    - `base(n, b)`: Formats integer `n` in base `b` (`2 ≤ b ≤ 36`) (for example, `base(255, 16)` evaluates to `"FF"`).
    - `approx(x)`: Forces decimal floating-point conversion (for example, `approx(1/3)` evaluates to `0.333333`).
    - `float(x)`: Forces decimal floating-point conversion (for example, `float(1/4)` evaluates to `0.25`).
    - `frac(x)`: Forces irreducible fraction representation (for example, `frac(0.75)` evaluates to `3/4`).
  - **Trigonometry**:
    - Standard: `sin(x)` (for example, `sin(pi/6)` evaluates to `0.5`), `cos(x)` (for example, `cos(pi/3)` evaluates to `0.5`), `tan(x)` (for example, `tan(pi/4)` evaluates to `1`), `cot(x)` (for example, `cot(pi/4)` evaluates to `1`), `sec(x)` (for example, `sec(0)` evaluates to `1`), `csc(x)` (for example, `csc(pi/2)` evaluates to `1`). Exact values return for multiples of `π/12`.
    - Degrees conversion: `deg(x)` converts degrees to radians (for example, `deg(180)` evaluates to `pi`), `rad_to_deg(rad)` (for example, `rad_to_deg(pi)` evaluates to `180`), `to_deg(rad)` (for example, `to_deg(pi/2)` evaluates to `90`).
  - **Inverse Trigonometry**:
    - `asin(x)` (for example, `asin(1/2)` evaluates to `pi/6`), `arcsin(x)` (for example, `arcsin(1)` evaluates to `pi/2`).
    - `acos(x)` (for example, `acos(1/2)` evaluates to `pi/3`), `arccos(x)` (for example, `arccos(1)` evaluates to `0`).
    - `atan(x)` (for example, `atan(1)` evaluates to `pi/4`), `arctan(x)` (for example, `arctan(1)` evaluates to `pi/4`).
  - **Complex Numbers**:
    - Imaginary unit `i` (`i² = -1`).
    - `re(z)`: Real part (for example, `re(3 + 4*i)` evaluates to `3`).
    - `im(z)`: Imaginary part (for example, `im(3 + 4*i)` evaluates to `4`).
    - `conj(z)`: Complex conjugate (for example, `conj(3 + 4*i)` evaluates to `3-4*i`).
    - `arg(z)`: Complex argument (phase angle) (for example, `arg(1 + i)` evaluates to `pi/4`).
    - `polar(z)`: Polar coordinates `[r, θ]` for complex number `z` (for example, `polar(1 + i)` evaluates to `[sqrt(2), pi/4]`).
    - `cis(theta)`: Euler formula `cos(θ) + i sin(θ)` (for example, `cis(pi)` evaluates to `-1`).
  - **Number Theory**:
    - `gcd(a, b)`: Greatest common divisor (for example, `gcd(12, 18)` evaluates to `6`).
    - `lcm(a, b)`: Least common multiple (for example, `lcm(4, 6)` evaluates to `12`).
    - `isprime(n)`: Deterministic primality test (for example, `isprime(17)` evaluates to `true`).
    - `factors(n)`: Prime factorization list (for example, `factors(60)` evaluates to `[2, 2, 3, 5]`).
    - `factor(n)`: Factor reduction of integers or polynomials (for example, `factor(12)` evaluates to `"2^2 * 3"`).
    - `comb(n, k)`: Binomial coefficient `C(n, k)` (for example, `comb(5, 2)` evaluates to `10`).
    - `perm(n, k)`: Permutations `P(n, k)` (for example, `perm(5, 2)` evaluates to `20`).
    - `factorial(n)`: Factorial `n!` (for example, `factorial(5)` evaluates to `120`).
    - `totient(n)`: Euler totient function `φ(n)` (for example, `totient(9)` evaluates to `6`).
    - `modexp(base, exp, mod)`: Modular exponentiation `base^exp (mod mod)` (for example, `modexp(2, 10, 1000)` evaluates to `24`).
    - `modinv(a, m)`: Modular multiplicative inverse (for example, `modinv(3, 11)` evaluates to `4`).
  - **Statistics & Collections**:
    - `min(...)`: Minimum of arguments or list (for example, `min(3, 1, 4)` evaluates to `1`).
    - `max(...)`: Maximum of arguments or list (for example, `max(3, 1, 4)` evaluates to `4`).
    - `mean(...)`: Arithmetic mean (for example, `mean(1, 2, 3, 4, 5)` evaluates to `3`).
    - `median(...)`: Median value (for example, `median(1, 3, 5, 7)` evaluates to `4`).
    - `std(...)`: Sample standard deviation (for example, `std(2, 4, 4, 4, 5, 5, 7, 9)` evaluates to `2.13809`).
    - `std_pop(...)`: Population standard deviation (for example, `std_pop(2, 4, 4, 4, 5, 5, 7, 9)` evaluates to `2`).
    - `sum(...)`: Sum of values or list (for example, `sum(1, 2, 3, 4)` evaluates to `10`).
    - `sum_range(start, end)`: Integer summation from `start` to `end` (for example, `sum_range(1, 100)` evaluates to `5050`).
    - `prod(...)`: Product of values or list (for example, `prod(2, 3, 4)` evaluates to `24`).
  - **Polynomials**:
    - `expand(expr)`: Expands polynomial products and powers (for example, `expand((x + 1)^2)` evaluates to `x^2 + 2*x + 1`).
    - `factor(poly)`: Factors polynomials over `ℚ` (for example, `factor(x^2 - 1)` evaluates to `(x - 1)*(x + 1)`).
    - `divide(p, q)`: Polynomial division returning quotient and remainder (for example, `divide(x^2 - 1, x - 1)` evaluates to `[x + 1, 0]`).
    - `pgcd(p, q)`: Greatest common divisor of polynomials (for example, `pgcd(x^2 - 1, x - 1)` evaluates to `x - 1`).
    - `roots(p)`: Finds rational and exact roots of polynomial `p` (for example, `roots(x^2 - 4)` evaluates to `[-2, 2]`).
    - `evaluate(poly, var, val)`: Evaluates polynomial at specific variable substitution (for example, `evaluate(x^2 + 1, x, 3)` evaluates to `10`).
  - **Geometry**:
    - `dist(p1, p2)`: Euclidean distance between points (for example, `dist([0, 0], [3, 4])` evaluates to `5`).
    - `midpoint(p1, p2)`: Midpoint between points (for example, `midpoint([0, 0], [4, 6])` evaluates to `[2, 3]`).
    - `slope(p1, p2)`: Slope of line passing through points (for example, `slope([0, 0], [2, 4])` evaluates to `2`).
    - `line(p1, p2)`: Linear equation of line (for example, `line([0, 0], [1, 2])` evaluates to `y = 2*x`).
    - `area_polygon(pts)`: Area of 2D polygon vertices (for example, `area_polygon([0, 0], [4, 0], [4, 3])` evaluates to `6`).
    - `heron(a, b, c)`: Heron's formula for triangle area from side lengths (for example, `heron(3, 4, 5)` evaluates to `6`).
    - `triangle_area(a, b, C)`: Triangle area from two sides and included angle (for example, `triangle_area(3, 4, pi/2)` evaluates to `6`).
    - `law_cos_side(a, b, C)`: Law of cosines finding third side (for example, `law_cos_side(3, 4, pi/2)` evaluates to `5`).
    - `law_cos_angle(a, b, c)`: Law of cosines finding included angle (for example, `law_cos_angle(3, 4, 5)` evaluates to `pi/2`).
    - `circ_area(r)`: Circle area `πr²` (for example, `circ_area(3)` evaluates to `9*pi`).
    - `circumference(r)`: Circle circumference `2πr` (for example, `circumference(3)` evaluates to `6*pi`).
    - `arc_length(r, theta)`: Circular arc length (for example, `arc_length(3, pi)` evaluates to `3*pi`).
    - `sector_area(r, theta)`: Circular sector area (for example, `sector_area(3, pi)` evaluates to `9*pi/2`).
  - **Linear Algebra**:
    - `det(A)`: Determinant of square matrix up to 10x10 (for example, `det([[1, 2], [3, 4]])` evaluates to `-2`).
    - `det3(a1,a2,a3, b1,b2,b3, c1,c2,c3)`: Fast 3x3 determinant from 9 scalar entries (for example, `det3(1,0,0, 0,1,0, 0,0,1)` evaluates to `1`).
    - `inv(A)`: Matrix inverse (for example, `inv([[1, 2], [3, 4]])` evaluates to `[[-2, 1], [1.5, -0.5]]`).
    - `transpose(A)`: Matrix transpose (for example, `transpose([[1, 2], [3, 4]])` evaluates to `[[1, 3], [2, 4]]`).
    - `mul(A, B)`: Matrix multiplication (for example, `mul([[1, 2]], [[3], [4]])` evaluates to `[[11]]`).
    - `rank(A)`: Matrix rank (for example, `rank([[1, 2], [2, 4]])` evaluates to `1`).
    - `trace(A)`: Matrix trace (for example, `trace([[1, 2], [3, 4]])` evaluates to `5`).
    - `solve_linear(A, b)`: Solves linear system `A·x = b` (for example, `solve_linear([[2, 1], [1, -1]], [5, 1])` evaluates to `[2, 1]`).
    - `rotation(theta)`: 2D rotation matrix for angle `θ` (for example, `rotation(pi/2)` evaluates to `[[0, -1], [1, 0]]`).
    - `eigen(A)`: Exact eigenvalues and eigenvectors for 2x2 matrices (for example, `eigen([[2, 0], [0, 3]])` evaluates to `[2, 3]`).
    - `dot(u, v)`: Vector dot product (for example, `dot([1, 2], [3, 4])` evaluates to `11`).
    - `cross(u, v)`: 3D vector cross product (for example, `cross([1, 0, 0], [0, 1, 0])` evaluates to `[0, 0, 1]`).
    - `norm(v)`: Euclidean vector norm (for example, `norm([3, 4])` evaluates to `5`).
    - `unit(v)`: Normalized unit vector (for example, `unit([3, 4])` evaluates to `[0.6, 0.8]`).
    - `proj(u, v)`: Vector projection of `u` onto `v` (for example, `proj([3, 4], [1, 0])` evaluates to `[3, 0]`).
    - `angle(u, v)`: Angle between vectors (for example, `angle([1, 0], [0, 1])` evaluates to `pi/2`).
  - **Probability Distributions**:
    - `binom_pmf(k, n, p)`: Binomial probability mass function (for example, `binom_pmf(2, 4, 1/2)` evaluates to `0.375`).
    - `binom_cdf(k, n, p)`: Binomial cumulative distribution function (for example, `binom_cdf(1, 4, 1/2)` evaluates to `0.3125`).
  - **Units & Constants**:
    - Physical units: `convert(val, from_unit, to_unit)` (for example, `convert(100, "cm", "m")` evaluates to `1 m`). Supported SI units: `m`, `s`, `kg`, `A`, `K`, `mol`, `cd`, `N`, `J`, `W`, `Pa`, `V`, `F`, `ohm`, `Hz`, and temperatures `celsius`, `degC`, `fahrenheit`, `degF`, `kelvin`, `K`.
    - Mathematical constants: `pi`, `e`. (Note: in algebraic contexts, single letters like `c` are treated as symbolic variables; for physical constants such as the speed of light, use `lookup("c")`).
    - Boolean constants: `true`, `false`.

### 5.2 `solve`: Symbolic Equation and Inequality Solver

The `solve` tool computes exact analytical solutions for algebraic equations and inequalities.

- **Invocation**:
  - Equations: `<|call:solve|> <equation>, <variable> <|/call|>` (for example, `solve "2*x + 3 = 7, x"` returns `x = 2 (<|r0|>)`)
  - Inequalities: `<|call:solve|> <inequality> <|/call|>` (for example, `solve "x^2 - 4 > 0"` returns `<|r0|> = (-inf, -2) U (2, inf)`)
- **Capabilities**:
  - **Linear Equations**: `a*x + b = c`.
  - **Quadratic Equations**: Exact closed-form solutions via discriminant analysis, returning exact surds.
  - **Higher-Degree Polynomials**: Solves cubics and quartics via the rational root theorem and factor reduction.
  - **Complex Solutions**: Returns "no real solutions" when no real roots exist. Complex roots are returned when the explicit `, complex` flag is appended to the invocation (for example, `solve "x^2 + 1 = 0, complex"` returns `x = -i (<|r0|>), x = i (<|r1|>)`).
  - **Linear Systems**: Solves systems of linear equations in multiple variables.
  - **Inequalities**: Univariate polynomial, rational, and absolute value inequalities (`<`, `<=`, `>`, `>=`). Solutions return in canonical interval notation:
    - Example: `x^2 - 4 > 0` returns `<|r0|> = (-inf, -2) U (2, inf)`
    - Example: `abs(x - 3) <= 2` returns `<|r0|> = [1, 5]`
    - Example: `x^2 >= 0` returns `<|r0|> = all real numbers`
    - Example: `x^2 < 0` returns `no solution`

### 5.3 `lua`: Embedded Algorithmic Sandbox

The `lua` tool executes dynamic algorithms in an embedded PUC-Rio Lua 5.4 runtime.

- **Invocation**: `<|call:lua|> <source_code> <|/call|>`
- **Execution Rules**:
  - Scripts MUST end with a return statement (for example, `return total` or `return a, b`).
  - Output functions such as `print()` are safe no-ops.
  - Multi-returns bind to a single register as a list.
  - Native 64-bit signed integers and IEEE 754 float64 arithmetic.
  - Injected math helper functions matching the `calc` tool: `gcd`, `lcm`, `isprime`, `factors`, `comb`, `perm`, `factorial`, `modinv`.
- **Security and Sandboxing**:
  - Permitted standard modules: `base`, `table`, `string`, `math`, `utf8`.
  - Prohibited standard modules: `os`, `io`, `debug`, `package`, `coroutine`.
  - Prohibited functions: `pcall`, `xpcall`, `load`, `loadstring`, `dofile`, `loadfile`, `require`, `collectgarbage`, `rawget`, `rawset`, `rawequal`, `string.dump`.
  - String metatable is locked (`__metatable = false`).
  - Deterministic instruction cap: Maximum 20,000,000 VM instructions per script execution, enforced via debug hooks (`LUA_MASKCOUNT`).
  - Example: `return 1+1` evaluates to `2`.
  - Deterministic memory cap: Maximum 10 MB heap allocation via custom allocator (`lua_Alloc`).
  - No system clock access or non-deterministic entropy.

### 5.4 `str`: Multibyte Rune-Aware String Engine

The `str` tool provides UTF-8 rune-accurate string manipulations.

- **Invocation**: `<|call:str|> <function>(<args>) <|/call|>`
- **Functions**:
  - `len(s)`: UTF-8 rune length (for example, `len("café")` evaluates to `4`).
  - `slice(s, start, end)`: 0-indexed rune slicing with bounds checking (for example, `slice("hello", 1, 3)` evaluates to `"el"`).
  - `reverse(s)`: Rune-preserving string reversal (for example, `reverse("hello")` evaluates to `"olleh"`).
  - `upper(s)`: Case conversion to uppercase (for example, `upper("hello")` evaluates to `"HELLO"`).
  - `lower(s)`: Case conversion to lowercase (for example, `lower("HELLO")` evaluates to `"hello"`).
  - `trim(s)`: Strips surrounding whitespace (for example, `trim("  hello  ")` evaluates to `"hello"`).
  - `contains(s, substr)`: Returns boolean indicator (for example, `contains("hello", "ell")` evaluates to `true`).
  - `count(s, substr)`: Counts non-overlapping occurrences (for example, `count("banana", "an")` evaluates to `2`).
  - `count_vowels(s)`: Counts case-insensitive ASCII vowels (for example, `count_vowels("hello world")` evaluates to `3`).
  - `find(s, substr)`: Returns 0-based rune index of first occurrence, or -1 (for example, `find("hello", "ll")` evaluates to `2`).
  - `replace(s, old, new)`: Substitutes occurrences (for example, `replace("hello", "l", "w")` evaluates to `"hewwo"`).
  - `to_hex(s)`: Uppercase hexadecimal representation of bytes (for example, `to_hex("abc")` evaluates to `"616263"`).
  - `from_hex(hex_str)`: Decodes hexadecimal bytes into UTF-8 string (for example, `from_hex("616263")` evaluates to `"abc"`).
  - `to_bin(n)`: Binary string representation of integer `n` (for example, `to_bin(10)` evaluates to `"1010"`).
  - `to_ascii(s)`: Returns the Unicode rune code point of a character (for example, `to_ascii("é")` evaluates to `233`).
  - `is_palindrome(s)`: Boolean test whether rune sequence is palindromic (for example, `is_palindrome("racecar")` evaluates to `true`).

### 5.5 `date`: Civil Calendar Arithmetic Engine

The `date` tool handles Gregorian civil calendar operations.

- **Invocation**: `<|call:date|> <function>(<args>) <|/call|>`
- **Clock Injection & Determinism**: Invocations of `now()` and `now("<IANA zone>")` read the host wall clock by default, making `now` the **only** harness call that reads the host clock when no explicit clock is supplied. To enforce 100% deterministic reproducibility across platforms, the clock can be overridden via:
  1. A request-level field in JSON-lines mode: `"clock": "2026-10-10T23:42:00Z"` (scoped to that single request).
  2. A CLI flag: `--clock <timestamp>` or environment variable `RECKONER_HARNESS_CLOCK` (overrides the default clock for the process).
  Time-zone names resolve through Go's `time.LoadLocation`, which reads the host's zoneinfo files first and falls back to the embedded `time/tzdata` copy, so results can differ only when the host's timezone data is out of date.
- **Functions**:
  - `now()`: Returns current timestamp string in UTC formatted as `"YYYY-MM-DD HH:MM MST"` (for example, `now()` evaluates to `"2026-10-10 23:42 UTC"`).
  - `now(timezone)`: Returns current timestamp string in the specified IANA timezone formatted as `"YYYY-MM-DD HH:MM MST"` (for example, `now("America/New_York")` evaluates to `"2026-10-10 19:42 EDT"`).
  - `add(date, count, unit)`: Adds days, weeks, months, or years. Clamps month ends cleanly (for example, `add("2026-10-10", 7, "days")` evaluates to `2026-10-17 (Saturday)`).
  - `diff(date1, date2)`: Signed civil day difference (`date2 - date1`) (for example, `diff("2026-10-10", "2026-10-17")` evaluates to `7`).
  - `diff_weeks(date1, date2)`: Full weeks difference between dates (for example, `diff_weeks("2026-10-10", "2026-10-24")` evaluates to `2`).
  - `day_of_week(date)`: Returns weekday name string (for example, `day_of_week("2026-10-10")` evaluates to `"Saturday"`).
  - `next_weekday(date, weekday)`: Advances strictly to the next specified weekday (for example, `next_weekday("2026-10-10", "Monday")` evaluates to `2026-10-12 (Monday)`).
  - `is_leap_year(year)`: Evaluates standard Gregorian leap year rule (for example, `is_leap_year(2024)` evaluates to `true`).
  - `days_in_month(year, month)`: Returns number of days in specified year and month (for example, `days_in_month(2024, 2)` evaluates to `29`).
  - `quarter(date)`: Returns calendar quarter (1..4) of specified date (for example, `quarter("2026-10-10")` evaluates to `4`).
- **Validation**: Nonexistent dates (such as `2025-02-30`) MUST return `ErrInvalidDate`. Unknown timezones MUST return `ErrSyntax`.

### 5.6 `lookup`: Offline Scientific Reference Table

The `lookup` tool returns scientific constants and reference data from embedded offline tables (no network access).

- **Invocation**: `<|call:lookup|> lookup("<key>"[, "<field>"]) <|/call|>` (for example, `lookup("c")` evaluates to `299792458 m/s`)
- **Data Sets & Supported Fields**:
  - **CODATA 2022 Fundamental Physical Constants**:
    - Supported fields: `value` (for example, `lookup("c", "value")` evaluates to `299792458 m/s`), `unit` (for example, `lookup("c", "unit")` evaluates to `"m/s"`), `uncertainty` (for example, `lookup("c", "uncertainty")` evaluates to `"exact"`).
  - **PubChem Periodic Table of Elements (1..118)**:
    - Supported fields: `name`, `symbol`, `atomic_number`, `z`, `number`, `atomic_weight` (for example, `lookup("Fe", "atomic_weight")` evaluates to `55.84`), `atomic_mass`, `weight`, `mass`, `group`, `period`, `density`, `melting_point`, `mp`, `boiling_point`, `bp`, `electronegativity`, `first_ionization_energy`, `ionization_energy`, `ie`, `electron_configuration`, `config`.
  - **NASA & IAU Planetary and Solar System Parameters**:
    - Supported fields: `name`, `diameter`, `mass` (for example, `lookup("Earth", "mass")` evaluates to `5.97e24 kg`), `gravity`, `surface_gravity` (for example, `lookup("Jupiter", "surface_gravity")` evaluates to `23.1 m/s^2`), `radius`, `density`, `orbital_period`, `rotation_period`, `distance_from_sun`, `semi_major_axis` (for example, `lookup("Mars", "semi_major_axis")` evaluates to `228.0e6 km`), `distance_from_earth`, `perihelion`, `aphelion`, `eccentricity`, `escape_velocity`, `orbital_velocity`, `solar_irradiance`, `luminosity`, `GM`, `gm` (for example, `lookup("Sun", "gm")` evaluates to `1.3271244e20 m^3/s^2`), `gravitational_parameter`, `solar_mass_parameter`.
- **Resolution Rules**:
  1. Exact match on identifier or alias: Binds exact value or property to register.
  2. Ambiguous query: Returns top candidate matches without binding.
  3. Unknown query: Emits `ErrNotFound`.

### 5.7 `reg`: Explicit Register Operations

The `reg` tool allows explicit inspection and assignment of registers without evaluation:
- `set r<N> <value>`: Sets register `r<N>` to `<value>` (returns `r<N> = <value>`).
- `get r<N>` or `r<N>`: Retrieves the current value of register `r<N>`.

---

## 6. Context References, Anchors and Quotes

The context subsystem provides byte-exact citation, message indexing, and long-context searching without generating redundant tokens.

### 6.1 Deterministic Anchor Inserter (`anchor`)

The anchor inserter scans raw token sequences, inserting message delimiter tokens `<|m0|>`..`<|m255|>` and line delimiter tokens `<|L0|>`..`<|L1023|>`. Conforming implementations MUST adhere to the following exact token-level rules:

1. **Initial Anchors**:
   - Every message MUST begin with `<|mN|>`, where N = `message_id` mod 256.
   - Immediately following `<|mN|>`, `<|L0|>` MUST be inserted before the first content token.
2. **Start-of-Line Anchor**:
   - A line anchor `<|Lk|>` MUST be inserted immediately before the very first token following a newline token.
   - Anchors MUST NOT be shifted past leading whitespace tokens; the line anchor is placed directly at the line start before any leading spaces or tabs.
3. **Mid-Line Anchors (64-Token Rule)**:
   - If 64 or more tokens have elapsed since the last anchor (`tokens_since_anchor >= 64`) and the current token is a whitespace token, a mid-line anchor `<|Lk|>` MUST be inserted immediately before that whitespace token.
4. **Vocabulary Token Sets**:
   - **Newline Tokens**: The set of 151 tokens in the v2 vocabulary whose decoded string contains `"Ċ"` or `"\n"`.
   - **Whitespace Tokens**: The set of 17,890 tokens in the v2 vocabulary whose decoded string begins with `"Ġ"`, begins with `" "`, or contains `"\t"`.
5. **Message Token Continuation Past 1,024 Line Anchors**:
   - A single message token `<|mN|>` supports up to 1,024 line anchors (`<|L0|>`..`<|L1023|>`).
   - Line numbers MUST NOT clamp at `<|L1023|>`. When `cur_line_idx >= 1024`, the message continues under the next message token: N' = (N + 1) mod 256, emitting `<|mN'|>`, resetting line count to 0, and emitting `<|L0|>`.
6. **Target Masking Invariant**:
   - All inserted anchor tokens (`<|mN|>`, `<|Lk|>`) MUST have their training loss targets strictly masked to `-100` (`IGNORE_INDEX`).
   - Content tokens retain their original target values.
7. **Active Message Slot Recycling**:
   - The active message slots cycle through 256 entries (`<|m0|>`..`<|m255|>`). When a session exceeds 256 messages, active slots wrap modulo 256 for direct quoting, while full conversation history MUST remain stored chronologically and fully searchable via `find`.
8. **Anchor-Free Mode**:
   - Systems MAY operate in anchor-free mode, configured via session option (CLI flag `--no-anchors`, JSON API `{"anchors": false}`, or front-door `{"anchors": false}` / `{"anchor_context": false}`). Default is anchors enabled (`true`) in the context store; `serve` and `format_prompt` insert anchors into rendered prompts only on request (see 12.3).
   - In anchor-free mode, the engine MUST NOT insert `<|mN|>` message tokens or `<|Lk|>` line anchor tokens into message token streams or tool observation responses.
   - Messages are segmented into lines by newline delimiters and indexed into the exact and BM25 search indices.
   - When anchors are disabled, `quote` and `quote_mask` MUST return `ErrUnsupported: quote is not supported in anchor-free mode`.

### 6.2 Context Range Resolution and Receipts (`quote`)

The model quotes ranges of anchored context using the dedicated delimiter sequence:

```text
<|quote|><|mN|><|La|><|Lb|><|/quote|>
```

- **Validation**: The referenced message slot `<|mN|>` MUST be active in the context store. The line range MUST satisfy 0 ≤ a ≤ b ≤ `num_lines` - 1.
- **Observation Receipt Format**:
  - Single-line quote (a = b):
    ```text
    <|obs|>Quoted 1 line (La): "<content>"<|/obs|>
    ```
  - Multi-line quote (a < b):
    ```text
    <|obs|>Quoted N lines (La-Lb): "<first_line>" … "<last_line>"<|/obs|>
    ```
- **Provenance Metadata**: The quote engine extracts the exact original byte sequence and returns provenance structure: `message_id`, `message_token`, `start_line`, `end_line`, `num_lines`, `num_tokens`, `char_start`, and `char_end` (representing exact 0-indexed Unicode rune offsets into the source message text).

### 6.3 Constrained Decoding Next-Token Masking (`quote_mask`)

To eliminate syntax errors during quote generation, the harness exposes allowed next-token bitmasks through a 4-step state machine:
1. `step = "message"`: After `<|quote|>`, allows only currently active message tokens (`<|m0|>`..`<|m255|>`).
2. `step = "start_line"`: After `<|mN|>`, allows valid line anchors for that message (`<|L0|>`..`<|Lk|>`).
3. `step = "end_line"`: After start line `<|La|>`, allows line anchors satisfying b ≥ a.
4. `step = "close"`: After end line `<|Lb|>`, allows only `<|/quote|>`.

### 6.4 Long-Context Search and Extraction (`find` and `lines`)

Long-context exploration operates without re-encoding text tokens, using `<|call:find|>` (or CLI / JSON API):

- **Search Engine Architecture (`find`)**:
  - Invocation: `<|call:find|>find("<needle>"[, <msg_id>])<|/call|>`
  - The search pipeline MUST execute in two stages:
    1. **Tier 1 (Normalized Exact Substring Search)**: The engine searches for exact occurrences of the normalized query within the normalized text of each line. If one or more lines match, the engine MUST return up to 3 excerpts formatted with the `[exact]` label, skipping Tier 2 entirely.
    2. **Tier 2 (BM25 Lexical Ranking Fallback)**: If and only if Tier 1 yields zero matching lines, the engine MUST score all candidate lines using the Okapi BM25 algorithm. Top scoring lines with score > 0 are returned formatted with the `[ranked]` label. If all lines score 0, the engine MUST return `<|obs|>no matches<|/obs|>`.
  - **Normalization Contract**: Both the query string and all indexed context lines MUST undergo identical normalization prior to matching or tokenization:
    1. *Unicode Normalization*: Apply Unicode Normalization Form KC (NFKC).
    2. *Punctuation Folding*:
       - Curly single quotation marks (`‘`, `’`, `‚`, `‛`) MUST be mapped to ASCII apostrophe `'` (`0x27`).
       - Curly double quotation marks (`“`, `”`, `„`, `‟`) MUST be mapped to ASCII quote `"` (`0x22`).
       - Dash variants (hyphen `‐` U+2010, non-breaking hyphen `‑` U+2011, figure dash `‒` U+2012, en-dash `–` U+2013, em-dash `—` U+2014, horizontal bar `―` U+2015, and mathematical minus `−` U+2212) MUST be mapped to ASCII hyphen-minus `-` (`0x2D`).
    3. *Thousands Separators*: Commas immediately preceded and followed by ASCII digits (`0-9`) MUST be removed (e.g. `1,250` becomes `1250`).
    4. *Whitespace Collapsing*: Any contiguous run of Unicode whitespace characters MUST be collapsed to a single ASCII space ` ` (`0x20`), and leading/trailing whitespace MUST be trimmed.
    5. *Case Folding*: All characters MUST be converted to lowercase using Unicode full case folding.
  - **BM25 Lexical Ranking Formulation**:
    - *Chunk Definition*: Chunks MUST be individual lines of anchored messages in scope (excluding derived search result messages).
    - *Tokenization*: The normalized line text MUST be segmented into tokens defined as contiguous sequences of Unicode letters (`unicode.IsLetter`), Unicode digits (`unicode.IsDigit`), and underscore `_`, plus a period `.` that sits immediately between two Unicode digits (so identifiers like `ch_9812`, version numbers like `v1.2.3`, and decimal numbers like `3.14` remain single tokens). No stemming or stop-word removal is performed.
    - *Scoring Parameters*: `k_1 = 1.2`, `b = 0.75`.
    - *IDF Formula*:
      $`\text{IDF}(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)`$
      where `N` is the total number of lines across all messages in scope, and `n(t)` is the number of lines containing term `t`.
    - *Line Score*:
      $`\text{Score}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}`$
      where `|D|` is the token count of the line, `avgdl` is the average token count across all lines in scope, and `f(t, D)` is the term frequency of `t` in line `D`. All computations MUST use IEEE-754 `float64` precision.
    - *Ranking and Tie-Breaking*: Candidate lines are sorted by score descending. Ties MUST be broken deterministically by earlier message (`turn_id` ascending), then earlier line index (`line_idx` ascending). Up to 3 top chunks with score > 0 are selected.
  - **Window Expansion and Overlap Merging**:
    - For each selected matching line, an excerpt window of ±2 surrounding lines (`[max(0, line - 2), min(num_lines - 1, line + 2)]`) within that message segment is extracted.
    - Overlapping or adjacent line ranges within the same message MUST be merged into contiguous segments.
  - **Output Formatting and Labels**:
    - Extracted lines MUST preserve their exact original (unnormalized) text.
    - **Anchors-Enabled Mode (Default)**:
      - The first line of each excerpt MUST prepend the match mode label immediately after the line anchor token:
        - Exact match excerpt: `<|Lk|>[exact] <line_content>`
        - Ranked match excerpt: `<|Lk|>[ranked] <line_content>`
        Subsequent lines within the excerpt MUST NOT receive the prefix label.
      - Multiple excerpts MUST be separated by a blank line with its own line anchor (`<|Lk|>\n`).
      - The entire excerpt block is anchored as a new message slot (`<|mK|>`) and returned as `<|obs|><|mK|><|L0|>...<|/obs|>`.
    - **Anchor-Free Mode**:
      - When anchors are disabled, excerpts are formatted as plain text without `<|mN|>` or `<|Lk|>` tokens.
      - The first line of each excerpt MUST begin with `[exact] <line_content>` or `[ranked] <line_content>`. Subsequent lines retain their exact original text.
      - Multiple excerpts are separated by a plain blank line (`\n\n`).
      - The excerpt text is returned directly as `<|obs|><plain_text_excerpts><|/obs|>`.
    - If no matches exist, returns `<|obs|>no matches<|/obs|>`.
  - **Scope and Invariants**:
    - Optional `msg_id` restricts search to that specific message. Derived search result messages MUST be excluded from unscoped search.
    - The context engine MUST maintain an incremental index of document frequencies, chunk lengths, and postings lists updated upon message ingestion, ensuring query evaluation executes without full context re-indexing.
- **Line Range Extraction (`lines`)**:
  - Invocation: `<|call:find|>lines(<msg_id>, <start_line>, <end_line>)<|/call|>`
  - Extracts lines `a..b` from active message slot `<|mN|>`.
  - In anchors-enabled mode, anchors the extracted lines as a new message slot and returns `<|obs|><|mK|><|L0|>...<|/obs|>`.
  - In anchor-free mode, returns the extracted lines as plain text without anchor tokens: `<|obs|><line_a>\n...\n<line_b><|/obs|>`.

### 6.5 Context Session Reset (`context_reset`)

Resets active message slots, conversation history, and quote masks to empty initial state.

---

## 7. Error Handling Contract

### 7.1 Standard Error Codes

All errors emitted by the harness conform to the standard error code table below:

| Error Code | Meaning |
| :--- | :--- |
| `ErrAmbiguous` | Ambiguous reference in lookup or resolution. |
| `ErrArgs` | Invalid arguments or argument count passed to tool. |
| `ErrDimension` | Matrix or vector dimension mismatch. |
| `ErrDivZero` | Division or modulo by zero. |
| `ErrDomain` | Mathematical domain error (such as `ln(0)` or `log(-1)`). |
| `ErrInfiniteSolutions` | System of equations has infinitely many solutions. |
| `ErrInternal` | Internal harness panic recovery or unexpected failure. |
| `ErrInvalidDate` | Invalid Gregorian calendar date format or nonexistent date. |
| `ErrInvalidUnit` | Unrecognized or malformed physical unit string. |
| `ErrLua` | Lua runtime error, memory limit exceeded, or prohibited operation. |
| `ErrNoSolution` | System of equations or inequality has no solution. |
| `ErrNotFound` | Requested element, constant, astronomical entity, or message slot not found. |
| `ErrOutOfBounds` | Line anchor, register index, or string slice index out of bounds. |
| `ErrOverflow` | Numeric overflow, infinite float, or value exceeding precision bounds. |
| `ErrRegisterBounds` | Register index outside permitted 0..15 range. |
| `ErrSyntax` | Parsing failure or malformed expression syntax. |
| `ErrTimeout` | Tool execution exceeded the 2-second dispatcher deadline. |
| `ErrTooLong` | Input sequence exceeds maximum permitted length. |
| `ErrTypeMismatch` | Operation attempted on incompatible value types. |
| `ErrUnitMismatch` | Incompatible physical units in arithmetic operation. |
| `ErrUnsetRegister` | Attempt to read from an uninitialized register. |
| `ErrUnsupported` | Tool or operation not supported by runtime configuration. |
| `ErrWorkLimitExceeded` | Evaluation exceeded bit-budget or iteration cap. |

### 7.2 Zero Panic Invariant

The harness MUST NOT crash or terminate abruptly on any input. All dispatch paths (CLI, stdio streaming, HTTP front door) MUST be wrapped in panic recovery handlers. In the event of an unhandled runtime error or malformed input, the harness MUST return an `ErrInternal` error response and restore state cleanly.

---

## 8. Determinism Properties

A conforming harness implementation MUST enforce 100% bit-identical determinism:
1. **Zero System Entropy**: The execution engine does not access system randomness, /dev/urandom, or random seed generation.
2. **Injectable Clock Determinism**: All date arithmetic (`add`, `diff`, `diff_weeks`, `day_of_week`, `next_weekday`, `is_leap_year`, `days_in_month`, `quarter`) operates exclusively on explicit calendar inputs. Only `date` `now()` reads the host wall clock by default, and is 100% deterministic when a clock timestamp is injected via `--clock`, `RECKONER_HARNESS_CLOCK`, or the request `"clock"` field (Section 5.5).
3. **Pure Functions**: For identical input strings and register states, the harness MUST produce byte-identical observation output strings.

---

## 9. Execution Interfaces

### 9.1 Single-Shot CLI Invocation

Conforming binaries accept single-shot CLI execution:

```bash
$ reckoner-harness calc '35 * 12 + 14'
<|r0|> = 434
```

```bash
$ reckoner-harness str 'len("café")'
<|r0|> = 4
```

```bash
$ reckoner-harness date 'add("2026-10-10", 7, "days")'
<|r0|> = 2026-10-17 (Saturday)
```

```bash
$ reckoner-harness lookup 'lookup("c")'
<|r0|> = 299792458 m/s
```

```bash
$ reckoner-harness solve '2*x + 3 = 7, x'
x = 2 (<|r0|>)
```

```bash
$ reckoner-harness lua 'return 1+1'
<|r0|> = 2
```

#### CLI Invocation Semantics

1. **Argument Joining**: The CLI joins extra command-line arguments into the tool payload using space delimiters. For example, `reckoner-harness calc 1 + 1` evaluates `1 + 1` yielding `2`; if a quoted expression is followed by extra arguments such as `calc "1+1" extra`, the combined string `1+1 extra` fails parsing and returns `ErrSyntax`.
2. **Clock Argument Stripping**: When passing `--clock <timestamp>` or `--clock=<timestamp>`, the flag and its value are stripped from the argument list before dispatching the payload to the tool engine.
3. **Strict Arity on `now()`**: The `date now()` tool accepts zero arguments (`now()`) or exactly one timezone string argument (`now("America/New_York")`). Supplying two or more arguments returns `ErrSyntax`.

### 9.2 Stdin/Stdout Streaming JSON Lines Protocol

For local evaluation pipelines, test runners, and subprocess wrappers, the harness maintains a persistent process reading newline-delimited JSON commands from standard input and emitting newline-delimited JSON responses to standard output.

- **Request Format (JSON line sent to harness stdin)**:
  ```json
  {"tool": "calc", "payload": "35 * 12 + 14"}
  ```
  *(The harness also accepts `"tool_name"` as an alias for `"tool"`).*

- **Response Format (JSON line emitted by harness stdout)**:
  ```json
  {"success": true, "result": "<|r0|> = 434", "elapsed_us": 6}
  ```

- **Error Response Format**:
  ```json
  {"success": false, "result": "", "error": "ErrDivZero: division by zero", "elapsed_us": 4}
  ```

- **Context Tool JSON Requests & Responses**:
  - `anchor`:
    ```json
    {"tool": "anchor", "tokens": [25049, 12511], "targets": [-100, 12511], "store": true}
    ```
    Response:
    ```json
    {"success": true, "result": "{\"next_message_id\":1,\"targets\":[-100,-100,-100,12511],\"tokens\":[32896,33152,25049,12511]}", "tokens": [32896, 33152, 25049, 12511], "targets": [-100, -100, -100, 12511], "next_message_id": 1, "elapsed_us": 25}
    ```
  - `quote`:
    ```json
    {"tool": "quote", "payload": "<|quote|><|m0|><|L0|><|L0|><|/quote|>"}
    ```
    Response:
    ```json
    {"success": true, "result": "Invoice", "receipt": "<|obs|>Quoted 1 line (L0): \"Invoice\"<|/obs|>", "text": "Invoice", "tokens": [25049, 12511], "provenance": {"message_id": 0, "message_token": "<|m0|>", "start_line": 0, "end_line": 0, "num_lines": 1, "num_tokens": 2, "char_start": 0, "char_end": 7}, "elapsed_us": 5}
    ```
  - `quote_mask`:
    ```json
    {"tool": "quote_mask", "prefix": ["<|quote|>", "<|m0|>"]}
    ```
    Response:
    ```json
    {"success": true, "result": "", "allowed_tokens": ["<|L0|>"], "allowed_token_ids": [33152], "step": "start_line", "elapsed_us": 1}
    ```
  - `find`:
    ```json
    {"tool": "find", "payload": "find(\"invoice\")"}
    ```
    Response:
    ```json
    {"success": true, "result": "<|m1|><|L0|>[exact] Invoice", "tokens": [32897, 33152, 186, 20162, 188, 348, 25049, 12511], "num_matches": 1, "elapsed_us": 12}
    ```
  - `lines`:
    ```json
    {"tool": "lines", "payload": "lines(0, 0, 0)"}
    ```
    Response:
    ```json
    {"success": true, "result": "<|m2|><|L0|>Invoice", "tokens": [32898, 33152, 25049, 12511], "num_matches": 1, "elapsed_us": 4}
    ```
  - `context_reset`:
    ```json
    {"tool": "context_reset"}
    ```

- **Session Reset**:
  Sending `{"tool": "reset"}` resets both the 16 circular registers and the context reference store to uninitialized state.

- **Dispatcher Timeout & Worker Isolation**:
  If a tool execution exceeds the 2-second dispatcher deadline, the harness MUST return an `ErrTimeout` response immediately. The abandoned background worker continues to execute until its internal step limit or panic recovery completes. The dispatcher executes register auto-binding strictly upon timely completion before the deadline; timed-out workers never commit register bindings or mutate store state. Note that instantaneous synchronous operations (`lookup`, `reg`, `anchor`, and `reset`) run directly on the dispatcher thread outside the worker timeout backstop.

Supported tool values: `"calc"`, `"solve"`, `"lua"`, `"str"`, `"date"`, `"lookup"`, `"reg"`, `"fncall"`, `"anchor"`, `"quote"`, `"quote_mask"`, `"find"`, `"lines"`, `"format_prompt"`, `"tokenize"`, `"encode"`, `"session_config"`, `"profile"`, `"context_reset"`, and `"reset"`.

---

## 10. Front-Door Serving Architecture

*Status: implemented in `reckoner-harness serve` (1.1.3) for both the embedded Reckoner 36,864-token vocabulary (`reckoner-1.4b` and no-profile) and external profile-configured Hugging Face tokenizers such as the 248K Qwen3.5 vocabulary (`reckoner-q2b`). The SGLang in-path integration ships in this repository under `integrations/sglang/`. llama.cpp is not supported in this release.*

### 10.1 Core Principle: Harness as the Front Door

In this serving architecture, **the Reckoner Harness is the front door, and inference engines only generate tokens**. Tool execution, context management, prompt templating, and OpenAI API compliance are handled exclusively by the harness. The underlying inference engine (such as SGLang, vLLM, llama.cpp, or custom PyTorch workers) does not run tool-call parsers, web servers, or external dispatch loops.

```
+-------------------------------------------------------------+
|                      Client Application                     |
|         POST /v1/chat/completions (OpenAI Compatible)       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                Reckoner Harness Front Door                  |
|  - Render Chat Template (System, User, Assistant turns)     |
|  - Run Anchor Inserter on context (<|mN|>, <|Lk|>)          |
|  - Enforce Reasoning Effort & Turn Limits                   |
|  - Manage Session Context & Circular Registers              |
+-------------------------------------------------------------+
         |                                           ^
         | Token IDs In                              | Token IDs Out
         v                                           | (Streaming)
+-------------------------------------------------------------+
|                   Inference Engine Core                     |
|          (Autoregressive Token Generation Only)             |
|                                                             |
|   Decodes tokens until <|/call|> or <|/quote|> observed     |
+-------------------------------------------------------------+
         |                                           ^
         +------------------+   +--------------------+
                            |   |
             Tool Dispatch  v   | Observation Prefill
+-------------------------------------------------------------+
|                In-Path Harness Execution                    |
|  - Microsecond-scale native tool execution (calc, lua)      |
|  - Resolve byte-exact quote citations & generate receipts   |
|  - Force observation tokens into engine sequence state      |
+-------------------------------------------------------------+
                              |
                              v (Post-Processing)
+-------------------------------------------------------------+
|                Client Response Delivery                     |
|  - Expand byte-exact quotes into response content           |
|  - Strip anchor control tokens (<|mN|>, <|Lk|>)             |
|  - Format JSON: content, reasoning_content, tool_calls      |
+-------------------------------------------------------------+
```

### 10.2 API Surface Specification

The harness exposes an OpenAI-compatible HTTP interface:
- **Endpoint**: `/v1/chat/completions`
- **Request Parameters**:
  - `model`: Target model identifier.
  - `messages`: Standard chat messages (`role`, `content`).
  - `temperature`, `top_p`, `max_tokens`: Generation controls.
  - `reasoning_effort`: Controls the `<|deliberate|>` gate mask, the deliberation token budget (`thinking_budget`), and, when enabled on the active model profile via `effort_line`, the `reasoning_effort: <name>` line in the `system` turn. (The `loop_cap` field in `conformance/effort_table.json` is the latent depth-loop cap `K_max`, which is reserved and not enforced; tool-call loop caps are governed by the session `max_tool_calls` limit, default `16`, rather than per-effort limits.)
  - `tools`: External agent tool schemas for `fncall`.
- **Response Fields**:
  - `choices[].message.content`: Clean assistant answer with internal quote references expanded and anchors stripped.
  - `choices[].message.reasoning_content`: Deliberation thought trace emitted after `<|deliberate|>` and before the next `<|ready|>` or `<|act|>` gate.
  - `choices[].message.tool_calls`: Formatted external function calls when external tools are dispatched via `fncall`.

#### Canonical `reasoning_effort` Table and Validation

Case-insensitive after trimming surrounding whitespace; an empty or omitted value resolves to `medium`:

| Canonical Name | Accepted Aliases | `gate_mask_deliberate` | `thinking_budget` (tokens) | `loop_cap` (reserved, not enforced) | Gate and Budget Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `none` | | `true` | `0` | `1` | `<\|deliberate\|>` masked at the turn-opening gate (`-inf` in native mode; `logit_bias: -100` on `<\|deliberate\|>` in `stop_and_continue` mode). |
| `minimal` | | `true` | `0` | `1` | Same gate mask and `0` thinking budget as `none`; when `effort_line` is `"none_only"` or `"all"`, renders `reasoning_effort: none` in the `system` turn. |
| `low` | `fast` | `false` | `256` | `2` | `<\|deliberate\|>` allowed; hard limit of `256` thinking tokens (forces `<\|ready\|>` at `thinking_tokens >= 256` in native mode). |
| `medium` (default) | `""` (omitted) | `false` | `1024` | `3` | `<\|deliberate\|>` allowed; hard limit of `1024` thinking tokens (forces `<\|ready\|>` at `thinking_tokens >= 1024` in native mode). |
| `high` | `careful`, `xhigh` | `false` | `4096` | `4` | `<\|deliberate\|>` allowed; hard limit of `4096` thinking tokens (forces `<\|ready\|>` at `thinking_tokens >= 4096` in native mode). |

- **HTTP Validation (`/v1/chat/completions`)**: Any unrecognized `reasoning_effort` string is rejected with HTTP `400 Bad Request` (`ErrArgs`, `text/plain`) before prompt rendering, backend dispatch, or streaming SSE headers (`text/event-stream`).
- **JSON IPC Validation (`format_prompt` / `render_prompt`)**: Accepts an optional `"reasoning_effort"` string field and returns `success: false` with an `ErrArgs` error on any unrecognized value.
- **In-Engine Logit Processor Fallback (`sglang_reckoner.tool_processor`)**: When `ReckonerToolLogitProcessor` is invoked directly against SGLang `/generate` without the Go front door, it resolves canonical names and aliases from the same table (the plugin uses a copy of `conformance/effort_table.json` kept in sync by a test); if an unrecognized `reasoning_effort` string is passed in `custom_params`, it logs a one-time warning per unknown name and falls back to `medium` (`gate_mask_deliberate: false`, `thinking_budget: 1024`).

### 10.3 In-Path Engine Integration Modes

1. **Native Mode (In-Path Decode Hook)**:
   - The front door passes the resolved canonical `reasoning_effort`, `thinking_budget`, and `gate_mask_deliberate` in `custom_params` to `ReckonerToolLogitProcessor`.
   - The engine decode loop halts when `<|/call|>` or `<|/quote|>` is emitted.
   - The harness executes the tool or resolves the quote in microseconds; the observation prefill then costs one short forward pass.
   - Observation tokens are appended directly to the existing sequence KV cache and recurrent GDN state (short prefill of 10–30 tokens).
   - Generation resumes immediately without exiting the engine scheduler or retokenizing context.
2. **Stop-and-Continue Mode (Compatibility Fallback)**:
   - For engines lacking in-path decode hooks (`--mode stop-and-continue` and `stop_and_continue` are both accepted), the engine ends generation at stop token `<|/call|>`.
   - The harness executes the tool and starts a continuation generation containing the accumulated sequence history plus observation tokens.
   - Prefix KV caching preserves prompt history across turns.
   - When `gate_mask_deliberate` is `true` (`reasoning_effort` is `none` or `minimal`), the harness sets `logit_bias: {"<deliberate_token_id>": -100}` in `sampling_params` on every generation round so `<|deliberate|>` cannot be sampled without requiring `custom_logit_processor`. When `gate_mask_deliberate` is `false` (`low`, `medium`, `high`, their aliases, or omitted/empty `""`), `stop_and_continue` mode runs without `custom_logit_processor` and logs a one-time warning to `stderr` per process stating that `thinking_budget` is not enforced in `stop_and_continue` mode.

---

## 11. Agent Function-Calling Layer (`fncall`)

Alongside the native atomic tools, the harness includes a function-calling subsystem for external agent tools.

### 11.1 Actions Supported

The `fncall` subsystem supports 5 primary operations:
- `render`: Renders JSON tool declarations into clean Python function signatures.
- `parse`: Parses a Pythonic function call string into structured tool name and argument values.
- `serialize`: Formats a tool name and argument map into canonical Python call syntax.
- `validate`: Validates a tool invocation against JSON schema parameter constraints.
- `grammar`: Generates EBNF or GBNF grammar constraints for constrained decoding.

### 11.2 Tool Catalog Declaration

External tools are provided to the model in the system turn using compact Pythonic declarations enclosed in `<|tools|>` delimiters:

```text
<|tools|>
def search(query: str, max_results: int = 5):
    """Search documents for relevant excerpts."""
<|/tools|>
```

CLI render invocation:

```bash
$ reckoner-harness fncall '{"op": "render", "tools": [{"name": "search", "description": "Search documents for relevant excerpts.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}}, "required": ["query"]}}]}'
def search(query: str, max_results: int = 5):
    """Search documents for relevant excerpts."""
```

### 11.3 Function Call Syntax

External calls use Python keyword argument syntax:

```text
<|call|>search(query="Mars atmosphere", max_results=3)<|/call|>
```

CLI parse invocation:

```bash
$ reckoner-harness fncall '{"op": "parse", "call": "search(query=\"Mars\")"}'
{"arguments":{"query":"Mars"},"name":"search"}
```

CLI serialize invocation:

```bash
$ reckoner-harness fncall '{"op": "serialize", "name": "search", "arguments": {"query": "Mars"}}'
search(query="Mars")
```

Parallel tool calls appear as consecutive `<|call|>` blocks. The model then yields generation, and observations return in matching sequence enclosed in `<|obs|>` delimiters.

### 11.4 Numeric Argument Serialization Rules

When serializing structured JSON arguments to Pythonic function calls (`op: "serialize"`), the harness MUST adhere to canonical Python `repr` numeric rules:
- **Integral Values and Large Floats**: In IEEE-754 double precision, floating-point numbers with absolute magnitude |x| ≥ 2⁵³ have no fractional component and are strictly integral; values at or above 10¹⁶ MUST be formatted as integer strings without decimal points or scientific notation (for example, `10000000000000000`, `12500000000000000`, `10000000000000002`). Any floating-point number whose fractional component is zero (`math.Floor(x) == x`) MUST be formatted without a trailing decimal point or zero (for example, `300.0` serializes as `300`, `0.0` and `-0.0` serialize as `0`).
- **Plain Decimals**: Floating-point numbers with a fractional component and absolute magnitude in the range 10⁻⁴ ≤ |x| < 10¹⁶ MUST be serialized as plain decimal strings without scientific notation (for example, `1234567.5`, `12345678.9`, `0.0001`).
- **Scientific Exponent Notation**: Scientific exponent notation applies only below 10⁻⁴ (|x| < 10⁻⁴); floating-point numbers in this range MUST be serialized in lowercase scientific notation (for example, `1e-05`, `1.5e-05`).
- **Arbitrary Precision Integers**: Large integers exceeding 53-bit IEEE-754 precision (> 2⁵³ - 1, such as `9007199254740993` or 30-digit integers) MUST be parsed and emitted without precision loss.
- **Rejection of Non-Standard JSON Literals**: In compliance with RFC 8259 JSON standards, non-standard numeric literals (`NaN`, `Infinity`, `-Infinity`) in JSON payloads MUST be rejected with `ErrSyntax`.

---

## 12. Model Profiles

*Status: implemented in 1.1.3 (`profiles/<name>.json`, embedded in the binary and loadable from disk via `--profile <name|path>`).*

Different models in the Reckoner family use different default system prompts, graded-evaluation answer instructions, and context-anchor settings. Instead of hardcoding model-specific prompt rules in external scripts, the harness reads declarative JSON model profiles.

### 12.1 Profile Schema

A model profile is a UTF-8 JSON object (`profiles/<name>.json`) with these fields:

```json
{
  "name": "reckoner-q2b",
  "default_system": "You are Reckoner-Q2B, a model by Maelie Labs.",
  "graded_answer_instruction": "Put your final answer within \\boxed{}.",
  "anchors": false,
  "effort_line": "off",
  "tokenizer": "tokenizer.json",
  "newline_token_id": 198,
  "control_tokens": {
    "<|endoftext|>": 248044,
    "<|pad|>": 248070,
    "<|turn:user|>": 248071,
    "<|turn:agent|>": 248072,
    "<|/turn|>": 248073,
    "<|turn:system|>": 248074,
    "...": 248111
  },
  "notes": "Reckoner-Q2B (2B dense hybrid) runs in anchor-free mode with a default identity system prompt."
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | `string` (required) | Canonical profile identifier. |
| `default_system` | `string` or `null` | Default system turn text inserted when a request has no `system` turn. When `null`, no system turn is inserted. |
| `graded_answer_instruction` | `string` or `null` | Instruction appended to the last user turn (`"\n\n" + graded_answer_instruction`) when a request is explicitly marked graded. |
| `anchors` | `boolean` | Whether context reference anchors (`<\|mN\|>`, `<\|Lk\|>`) and `<\|quote\|>` are enabled by default for this model. |
| `effort_line` | `string` (optional) | Controls whether a `reasoning_effort: <name>` line is written into the `system` turn: `"off"` (default when omitted or `""`), `"none_only"`, or `"all"`. Any other value is rejected at profile load time with `ErrArgs`. |
| `tokenizer` | `string` (optional) | Tokenizer source: `"embedded"` (or omitted) for the built-in 36,864-token 1.4B BPE tokenizer, or a file path (default `"tokenizer.json"`) to a Hugging Face `tokenizer.json` file. |
| `newline_token_id` | `integer` (optional) | Single-byte newline (`\n`) token ID (`326` in the embedded 1.4B tokenizer; `198` in the Qwen3.5 248K tokenizer). |
| `control_tokens` | `object` (optional) | Map from each of the 43 required control token strings (`<\|endoftext\|>`, `<\|pad\|>`, `<\|turn:user\|>`..`<\|r15\|>`) to its integer token ID in the profile's vocabulary. |
| `notes` | `string` | Human-readable notes describing the model family and profile conventions. |

### 12.2 Built-In Profiles

The harness embeds two standard profiles in the binary via `go:embed`:

| Profile | `default_system` | `graded_answer_instruction` | `anchors` | `effort_line` | `tokenizer` | `newline_token_id` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `reckoner-1.4b` | `null` | `"Put your final answer within \\boxed{}."` | `true` | `"off"` | embedded (36,864 vocab) | `326` |
| `reckoner-q2b` | `"You are Reckoner-Q2B, a model by Maelie Labs."` | `"Put your final answer within \\boxed{}."` | `false` | `"off"` | `"tokenizer.json"` (248K vocab) | `198` |

### 12.3 Selection and Precedence Rules

1. **Selection and Clearing (`"profile": ""`)**: A profile is selected via `--profile <name|path>` on the CLI and on `reckoner-harness serve`, or per request via the `"profile"` field in JSON IPC (`session_config`, `format_prompt`, `anchor`, `find`) and `/v1/chat/completions`. If `<name|path>` matches a path on disk (or contains a path separator or `.json` suffix), the file is loaded from disk; otherwise it is resolved from the embedded `profiles/<name>.json` table. Passing `"profile": ""` in a JSON request clears an active profile (for the session in `session_config`, or for that request) and resets the context store's anchors to the CLI default (`!no_anchors`).
2. **No Profile**: When `--profile` is omitted, no default system turn is inserted, no graded instruction is appended, and the context store's anchors default to `true` (prompt rendering in `serve` and `format_prompt` inserts anchors only when asked: `"anchor_context": true` or `"anchors": true` in the request, or `serve --anchor-context`). *(Note on `reckoner-harness serve` CLI flag parsing: `serve` recognizes `--anchor-context` as a boolean flag without `=true`, whereas `--anchor-context=true` is ignored; `--anchors` and `--no-anchors` are read by the global flag parser before `serve` sees its arguments.)*
3. **Anchor Override and Precedence (`effective_anchors`)**: When a profile is active, effective anchor activation follows `effective_anchors = profile.anchors && !no_anchors && (explicit_anchors != false)`. Any disablement (`profile.anchors == false`, `--no-anchors` / `"no_anchors": true`, or `--anchors=false` / `"anchors": false`) switches anchors off. In particular, a profile with `anchors: false` (`reckoner-q2b`) wins over any explicit `anchors: true`, `--anchors`, or `anchor_context: true` in stdin JSON IPC, `format_prompt`, and `serve` alike.
4. **System Turn Insertion**: When a profile with a non-null `default_system` is active and the incoming message list contains no turn with `role == "system"`, the front door prepends a `system` turn with `content: default_system` before rendering the canonical turn layout (`<|turn:system|>\n...<|/turn|>\n`). If the caller already supplies a `system` turn, `default_system` is not inserted.
5. **Graded Answer Instruction**: When a request is marked graded (`"graded": true` in the request JSON or `--graded` on the CLI) and the active profile has a non-empty `graded_answer_instruction`, the harness appends `"\n\n" + graded_answer_instruction` to the content of the last `user` turn. Interactive chat requests are ungraded by default (`graded: false`).
6. **Tokenizer and Control-Token Resolution (`--tokenizer <path>`, `--model-dir <dir>` / `--model-path <dir>` / `--model <dir>`)**: External tokenizer paths are consulted only when the active profile declares an external `tokenizer` (`profile.tokenizer` is non-empty and not `"embedded"`, such as `"tokenizer.json"` in `reckoner-q2b`). An external tokenizer profile MUST specify both `control_tokens` (all 43 required entries) and `newline_token_id`; omitting either, or specifying `control_tokens` or `newline_token_id` on an empty or `"embedded"` `tokenizer`, is rejected with `ErrArgs`. When no profile is active or the profile uses the embedded tokenizer (`reckoner-1.4b`), `RECKONER_TOKENIZER_PATH` and `--model-dir` / `--model-path` / `--model` are ignored and an explicit non-whitespace `--tokenizer` flag or JSON IPC `"tokenizer"` field is rejected with `ErrArgs` (while `"tokenizer": ""` on 1.4B / no-profile succeeds; HTTP `/v1/chat/completions` does not accept a per-request `tokenizer` field). In `session_config`, a stored session `--tokenizer` path is cleared whenever the profile name or disk profile file path (`SourcePath`) changes (or when the profile is cleared with `"profile": ""`). When `"tokenizer"` is explicitly supplied in `session_config`, `format_prompt`, or `tokenize` for an external-tokenizer profile, the harness rejects an empty or whitespace-only `"tokenizer"` value with `ErrArgs` and validates non-empty paths eagerly (`ErrArgs` on failure); when `"tokenizer"` is omitted in `session_config`, `session_config` updates the session profile and anchor policy immediately (`"OK"`) so non-tokenizing context and atomic operations (`anchor`, `quote`, `find`, `lines`) can run without loading the external `tokenizer.json`, and tokenizer-dependent operations (`format_prompt`, `tokenize`) resolve and validate the external tokenizer on call (returning `ErrArgs` if no tokenizer path resolves). When the profile uses an external tokenizer, the path is resolved in strict order without falling back to the current working directory or to the embedded 1.4B tokenizer: (1) `--tokenizer <path>` on the CLI or `"tokenizer"` in a local JSON IPC request (`session_config`, `format_prompt`, `tokenize`), where `<path>` may be a file or a directory (joining `profile.tokenizer`), (2) `RECKONER_TOKENIZER_PATH` environment variable (file or directory), (3) an absolute `profile.tokenizer` path if `profile.tokenizer` is itself an absolute path (overriding the directory steps below), (4) the model directory (`--model-dir <dir>`, `--model-path <dir>`, or `--model <dir>`) if that directory contains `profile.tokenizer`, and (5) the directory of the disk profile file (resolved to an absolute path when the profile was loaded from disk) if that directory contains `profile.tokenizer`. Otherwise, resolution fails with a clear `ErrArgs` error naming `--tokenizer`. At load time, the harness validates that all 43 required control tokens are present in `control_tokens` with distinct IDs matching `tokenizer.json` (`model.vocab` and `added_tokens`), that `newline_token_id` matches the single-token encoding of `"\n"`, and that `decode(encode(token))` round-trips every control token back to its exact string.
7. **Byte-Level BPE Pre-Tokenization, NFC Normalization, and Fail-Closed Pipeline Checks**: When loading an external Hugging Face `tokenizer.json` that configures NFC normalization (`normalizer.type == "NFC"`) and the Qwen3.5 `Split` (`Isolated`) + `ByteLevel` (`add_prefix_space: false`, `use_regex: false`) pre-tokenizer (as in `reckoner-q2b`), the harness fails closed on any unsupported normalizer, pre-tokenizer, `added_tokens` flags (requiring `lstrip: false`, `rstrip: false`, `single_word: false`, and `normalized: false` on every added token), `post_processor` (requiring `null` or `ByteLevel` with `add_prefix_space: false`, `use_regex: false`), `decoder` (requiring `ByteLevel` with `add_prefix_space: false`, `use_regex: false`), or BPE model settings (`byte_fallback: false`, empty `continuing_subword_prefix`/`end_of_word_suffix`, `unk_token: null`, `ignore_merges: false`), and supports both `["a", "b"]` pair arrays and `"a b"` space-delimited strings in `model.merges`. During `Encode(text)`, the tokenizer first segments registered `added_tokens` (`normalized: false`) by exact longest match from a first-byte lookup table, applies pure UAX #15 Unicode NFC normalization (without Go `golang.org/x/text/unicode/norm`'s Stream-Safe `U+034F` insertion on runs exceeding 30 non-starter combining marks, matching Rust `unicode-normalization` in Hugging Face `tokenizers`) to each non-added plain-text segment, and splits each normalized segment with the ordered 7-branch pre-tokenizer pattern `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` before applying byte-level BPE merges. When no profile is active or `reckoner-1.4b` is selected, the embedded 36,864-token tokenizer uses its existing pre-tokenizer without NFC normalization so 1.4B encoding remains byte-identical.
8. **Special-Token Escaping in Prompt Rendering, In-Path Tool Observations, and Raw Tokenization (`tokenize` / `encode`)**: Prompt rendering in `reckoner-harness serve` and `format_prompt`, as well as in-path tool observation encoding, enforces the same escaping rule across both 1.4B / no-profile execution (`reckoner-1.4b`) and external-tokenizer profiles (`reckoner-q2b`) so no tool observation body and no user or client text can inject a non-template control token ID into the token stream:
   - **Prompt Rendering and In-Path Tool Observation Escaping (Both 1.4B / No-Profile and External-Tokenizer Profiles)**: The Go harness replaces every literal `<|` with `<\|` and, when the active tokenizer declares non-`<|...|>` angle-bracket `added_tokens` (such as Qwen3.5's `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` → `<\think>`, `<\/think>`, `<\tool_call>`, `<\/tool_call>`, `<\tool_response>`, `<\/tool_response>`), escapes those tags across `system` `content` (preserving a leading `<|tools|>` and its matching `<|/tools|>` delimiter while escaping both the body inside `<|tools|>...<|/tools|>` and all text after `<|/tools|>`, or after `<|tools|>` if `<|/tools|>` is absent), the OpenAI `tools` catalog rendered by `fncall.Render` (escaping function descriptions, parameter descriptions, and other free-text schema fields inside the outer `<|tools|>\n...\n<|/tools|>` wrapper), `user` `content`, `assistant` `reasoning_content`, `assistant` plain `content` (when it does not begin with `<|ready|>`, `<|act|>`, `<|deliberate|>`, or `<|clarify|>`), and `tool` `content`. In the Python bridge (`sglang_reckoner.harness_bridge.escape_control_tokens`), which receives a `tokenizer_path` (or `RECKONER_TOKENIZER_PATH` environment variable) or explicit `added_tokens` list rather than a profile object, non-`<|...|>` angle-bracket `added_tokens` are loaded from the configured tokenizer path (failing closed by raising `FileNotFoundError` or `ValueError` if an explicit whitespace `tokenizer_path` or whitespace-only `RECKONER_TOKENIZER_PATH` is set or if a configured tokenizer path is missing, unreadable, or lacks an `"added_tokens"` list, while returning `[]` when no tokenizer path is configured). During in-path tool and quote execution (`reckoner-harness serve`, `ContextStore.Quote`, and `sglang_reckoner.tool_processor.ReckonerToolLogitProcessor._execute_tool_call` / `_execute_quote`), both 1.4B / no-profile and external-tokenizer / Qwen-vocab sessions encode every tool and quote observation through a single choke point (`EncodeToolObservation` in Go and `_encode_tool_observation` in Python) that preserves `<|obs|>` / `<|obs:err|>`, the leading `<|rN|> = ` register prefix on built-in tools (`calc`, `solve`, `str`, `date`, `lookup`, `lua`, `reg`, plus `(<|rN|>)` / `(<|rN|>, replaced ...)` annotations in `solve`), and `<|/obs|>` while escaping both `<|` and registered non-`<|...|>` angle-bracket `added_tokens` across the entire tool-produced body (success or error across all built-in and client tools, including `external`/`fncall` observations starting with `<|rN|> = `, `quote` receipts/errors, and `find`/`lines` results) without double-escaping `<\|`. Under this single body-escaping rule, `<|...|>` tags written by the harness embedded inside error observations (`ErrUnsetRegister: <|rN|> is empty`, `ErrNotFound: message token <|mN|> not found in active context`, `ErrOutOfBounds: line index L... out of bounds (message <|mN|> has ... lines: ...)`, `ErrSyntax: malformed quote syntax: <|quote|>...<|/quote|>`) and inside `reg list` entries after the first `<|rN|> = ` prefix encode those tags as escaped text (`<\|rN|>`, `<\|mN|>`, `<\|quote|>`, `<\|/quote|>`) on 1.4B / no-profile just as on `reckoner-q2b`. On 1.4B / no-profile `find` and `lines`, `rawFindTokens` (including `<|m0|>..<|m255|>` and `<|L0|>..<|L1023|>` context anchor tokens and `<|obs|><|/obs|>` on zero matches) are preserved when they contain only BPE text tokens and context anchor tokens, while any non-anchor control or reserved token ID (`< 128` or `> 34175`, such as a `find` or `lines` hit on the `<|tools|>...<|/tools|>` block in the `system` turn) is escaped to `<\|tools|>` / `<\|/tools|>` text (as on `reckoner-q2b`). For structured `assistant` `tool_calls`: on external-tokenizer profiles (`reckoner-q2b`), `tc.Function.Name` must either match a declared tool in `tools` (by original or sanitized name, serializing as its sanitized identifier) or be a valid ASCII identifier (`[A-Za-z_][A-Za-z0-9_]*`), and every argument key in `tc.Function.Arguments` must either match a declared property of that tool (serializing as its sanitized identifier) or be a valid ASCII identifier (`[A-Za-z_][A-Za-z0-9_]*`), failing closed with `ErrArgs` (HTTP `400`) otherwise, while multi-line `"""..."""` string argument bodies and the raw fallback when `tc.Function.Arguments` is not a JSON object are escaped with `escapeUserSystemTokens`; on 1.4B / no-profile, non-identifier tool names (`undeclared-tool`, `github.create-issue` without `tools`) and non-identifier argument keys (`extra-key`) are accepted and escaped with `escapeUserSystemTokens` (along with multi-line `"""..."""` string values) while non-object `tc.Function.Arguments` (`"not json"`, `null`, `[1,2]`) render `escapedName()`. Pre-formatted `assistant` `content` beginning with `<|ready|>`, `<|act|>`, `<|deliberate|>`, or `<|clarify|>` is validated on both 1.4B / no-profile and external-tokenizer profiles against the allowed and forbidden token sets derived from the active `TokenTable` and `tok.AddedTokens` (so pre-formatted `assistant` `content` containing or ending in `<|endoftext|>` or `<|/turn|>` is rejected with `ErrArgs` / HTTP `400` on 1.4B / no-profile as on `reckoner-q2b`):
     - **Allowed agent-turn tokens (34 control tokens on `reckoner-q2b`, plus `<|quote|>`, `<|/quote|>`, and context anchor tokens `<|m0|>..<|m255|>` / `<|L0|>..<|L1023|>` on 1.4B / no-profile when quote delimiters are enabled)**: mode tokens (`<|deliberate|>`, `<|act|>`, `<|ready|>`, `<|clarify|>`), tool-call delimiters (`<|call|>`, `<|/call|>`, `<|call:calc|>`, `<|call:solve|>`, `<|call:lua|>`, `<|call:str|>`, `<|call:date|>`, `<|call:lookup|>`, `<|call:find|>`, `<|call:tools|>`, `<|call:def|>`), observation delimiters (`<|obs|>`, `<|obs:err|>`, `<|/obs|>`), register tokens (`<|r0|>..<|r15|>`), and quote delimiters (`<|quote|>`, `<|/quote|>`) plus context anchor tokens (`<|m0|>..<|m255|>`, `<|L0|>..<|L1023|>`) when enabled on the profile.
     - **Forbidden tokens (rejected with `ErrArgs` / HTTP `400`)**: all non-agent control tokens (`<|endoftext|>`, `<|pad|>`, `<|turn:user|>`, `<|turn:agent|>`, `<|/turn|>`, `<|turn:system|>`, `<|tools|>`, `<|/tools|>`, `<|pause|>`), `<|quote|>` and `<|/quote|>` when the profile's `control_tokens` do not define them, and every `added_token` in `tok.AddedTokens` outside the allowed agent-turn set (including reserved tokens on 1.4B and all Qwen special/added tokens `248044..248069` on `reckoner-q2b`).
   - **Raw Tokenization (`tokenize` / `encode`)**: In the additive `"tokenize"` (or `"encode"`) JSON IPC action, no prompt-level escaping is applied; exact matches of registered `added_tokens` (including the 43 `<|...|>` control tokens and Qwen3.5 special tags such as `<tool_call>` and `<think>`) emit their single special-token IDs, while partial or unregistered lookalike strings (such as `<|r0`, `<|r16|>`, or `</|ready|>`) remain intact in the plain-text segment and are tokenized through normal BPE merges.
9. **Reasoning Effort System Line (`effort_line`)**: After `ApplyProfileToMessages` runs (so `default_system` and `graded_answer_instruction` are already in place), `ApplyEffortLineToMessages` optionally writes a `reasoning_effort: <name>` line into the `system` turn according to the profile's `effort_line` setting:
   - `"off"` (or omitted / `""`): Never writes a `reasoning_effort:` line into the `system` turn. Both built-in profiles (`reckoner-1.4b` and `reckoner-q2b`) set `"effort_line": "off"`.
   - `"none_only"`: Writes `reasoning_effort: none` into the `system` turn when the resolved canonical effort is `none` or `minimal`, and writes nothing for `low`, `medium`, or `high` (or their aliases).
   - `"all"`: Writes `reasoning_effort: <name>` (`none` for both `none` and `minimal`, `low`, `medium`, or `high`; aliases `fast` → `low`, `careful` → `high`, `xhigh` → `high`, empty → `medium`) into the `system` turn on every request.
   - **Placement and Idempotence**: If the message list has no `system` message after `ApplyProfileToMessages` (for example, `default_system` is `null` and the caller passed no `system` message), the harness inserts `{"role": "system", "content": "reasoning_effort: <name>"}` at index `0` (emitting `<|turn:system|>\nreasoning_effort: <name><|/turn|>\n`, or `<|turn:system|>\nreasoning_effort: <name>\n<|tools|>\n...\n<|/tools|><|/turn|>\n` when external `tools` are also passed in the request). If the message list already has a `system` message, only the first `system` message is modified: if its content already begins with `reasoning_effort:`, it is left unchanged (no second line is added); if its content is empty, it is set to `"reasoning_effort: <name>"` (with no trailing `\n`); otherwise (including when its content begins with `<|tools|>`), the line is placed first, before any `<|tools|>` block, as `"reasoning_effort: <name>\n" + existing_system` (preserving the `<|tools|>...<|/tools|>` control tokens only when the harness itself inserted that leading effort line). Because `ApplyEffortLineToMessages` runs before `RenderChatPromptWithTable`, the `reasoning_effort: <name>` line is part of message `0` in both `ContextStore` (message `0`, line `0`) and `RawMessagesTokens[0]` (`context_messages_tokens` in native mode's `custom_params`).

---

## 13. Planned Protocol Extensions

Planned additions to the protocol specification:

1. **Dynamic Catalog Search**:
   - Protocol token: `<|call:tools|> <query> <|/call|>`
   - Enables retrieving relevant external tool schemas dynamically from large tool catalogs.
2. **Dynamic Tool Definitions**:
   - Protocol token: `<|call:def|> <lua_tool_definition> <|/call|>`
   - Allows models to define new callable Lua tools dynamically during inference.
