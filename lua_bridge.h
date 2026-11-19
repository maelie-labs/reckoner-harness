#ifndef LUA_BRIDGE_H
#define LUA_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// run_lua_sandboxed executes a sandboxed Lua 5.4 script with:
// - 10 MB strict memory allocator ceiling
// - 20M VM instructions primary execution ceiling with 1s wall-clock backstop (hook-based)
// - Restricted libraries: base, table, string, math, utf8 only
// - Stripped unsafe functions: pcall, xpcall, load, loadstring, dofile, loadfile, require,
//   collectgarbage, rawget, rawset, rawequal, string.dump, protected string metatable
// - Injected math helpers: gcd, lcm, isprime, factors, comb, perm, factorial, modinv
// - Auto-injected registers: r0..r7
// Returns 0 on success (with out_result allocated via malloc), or non-zero on error (with out_err allocated).
#define REG_TYPE_AUTO   0
#define REG_TYPE_INT    1
#define REG_TYPE_FLOAT  2
#define REG_TYPE_BOOL   3
#define REG_TYPE_STRING 4

int run_lua_sandboxed(
    const char *script,
    int reg_count,
    const char **reg_names,
    const char **reg_vals,
    const int *reg_types,
    char **out_result,
    char **out_err
);

void free_lua_string(char *s);

#ifdef __cplusplus
}
#endif

#endif /* LUA_BRIDGE_H */
