#define MAKE_LIB
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

// Lua 5.4 core and safe libraries in a single compilation unit
#include "third_party/lua-5.4.9/src/lprefix.h"
#include "third_party/lua-5.4.9/src/lapi.c"
#include "third_party/lua-5.4.9/src/lcode.c"
#include "third_party/lua-5.4.9/src/lctype.c"
#include "third_party/lua-5.4.9/src/ldebug.c"
#include "third_party/lua-5.4.9/src/ldo.c"
#include "third_party/lua-5.4.9/src/ldump.c"
#include "third_party/lua-5.4.9/src/lfunc.c"
#include "third_party/lua-5.4.9/src/lgc.c"
#include "third_party/lua-5.4.9/src/llex.c"
#include "third_party/lua-5.4.9/src/lmem.c"
#include "third_party/lua-5.4.9/src/lobject.c"
#include "third_party/lua-5.4.9/src/lopcodes.c"
#include "third_party/lua-5.4.9/src/lparser.c"
#include "third_party/lua-5.4.9/src/lstate.c"
#include "third_party/lua-5.4.9/src/lstring.c"
#include "third_party/lua-5.4.9/src/ltable.c"
#include "third_party/lua-5.4.9/src/ltm.c"
#include "third_party/lua-5.4.9/src/lundump.c"
#include "third_party/lua-5.4.9/src/lvm.c"
#include "third_party/lua-5.4.9/src/lzio.c"

#include "third_party/lua-5.4.9/src/lauxlib.c"
#include "third_party/lua-5.4.9/src/lbaselib.c"
#include "third_party/lua-5.4.9/src/lmathlib.c"
#include "third_party/lua-5.4.9/src/lstrlib.c"
#include "third_party/lua-5.4.9/src/ltablib.c"
#include "third_party/lua-5.4.9/src/lutf8lib.c"

#include "lua_bridge.h"

#define MAX_HEAP_BYTES            (10 * 1024 * 1024) // 10 MB hard memory cap
#define MAX_VM_INSTRUCTIONS       20000000ULL        // 20M VM instructions primary limit
#define TIMEOUT_WALLCLOCK_MS      1000ULL            // 1 s generous wall-clock backstop
#define HOOK_STEP                 1000

// ---------------------------------------------------------
// Time utilities
// ---------------------------------------------------------

static uint64_t get_time_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

// Thread-local state for deterministic instruction and wall-clock execution hooks
static __thread uint64_t tl_instruction_count = 0;
static __thread uint64_t tl_deadline_ns = 0;

static void timeout_hook(lua_State *L, lua_Debug *ar) {
    (void)ar;
    tl_instruction_count += HOOK_STEP;
    if (tl_instruction_count > MAX_VM_INSTRUCTIONS) {
        luaL_error(L, "instruction budget exceeded");
    }
    if (tl_deadline_ns > 0 && get_time_ns() > tl_deadline_ns) {
        luaL_error(L, "execution deadline exceeded");
    }
}

// ---------------------------------------------------------
// Memory Limiter (custom lua_Alloc)
// ---------------------------------------------------------

typedef struct {
    size_t used_bytes;
    size_t max_bytes;
} MemLimiter;

static void *sandboxed_alloc(void *ud, void *ptr, size_t osize, size_t nsize) {
    MemLimiter *ml = (MemLimiter *)ud;
    if (nsize == 0) {
        if (ptr != NULL) {
            if (osize <= ml->used_bytes) {
                ml->used_bytes -= osize;
            } else {
                ml->used_bytes = 0;
            }
            free(ptr);
        }
        return NULL;
    }
    if (ml->used_bytes + nsize - osize > ml->max_bytes) {
        return NULL; // Allocation beyond cap raises "not enough memory"
    }
    void *new_ptr = realloc(ptr, nsize);
    if (new_ptr != NULL) {
        ml->used_bytes += (nsize - osize);
    }
    return new_ptr;
}

static int dummy_print(lua_State *L) {
    (void)L;
    return 0;
}

// ---------------------------------------------------------
// Unified Mathematical Helpers
// ---------------------------------------------------------

static int64_t c_gcd(int64_t a, int64_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) {
        int64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static int lua_helper_gcd(lua_State *L) {
    lua_Integer a = luaL_checkinteger(L, 1);
    lua_Integer b = luaL_checkinteger(L, 2);
    lua_pushinteger(L, c_gcd(a, b));
    return 1;
}

static int lua_helper_lcm(lua_State *L) {
    lua_Integer a = luaL_checkinteger(L, 1);
    lua_Integer b = luaL_checkinteger(L, 2);
    if (a == 0 || b == 0) {
        lua_pushinteger(L, 0);
        return 1;
    }
    int64_t g = c_gcd(a, b);
    int64_t abs_a = a < 0 ? -a : a;
    int64_t abs_b = b < 0 ? -b : b;
    lua_pushinteger(L, (abs_a / g) * abs_b);
    return 1;
}

static int c_isprime(int64_t n) {
    if (n < 2) return 0;
    if (n == 2 || n == 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int64_t i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

static int lua_helper_isprime(lua_State *L) {
    lua_Integer n = luaL_checkinteger(L, 1);
    lua_pushboolean(L, c_isprime(n));
    return 1;
}

static int lua_helper_factors(lua_State *L) {
    lua_Integer n = luaL_checkinteger(L, 1);
    lua_newtable(L);
    if (n <= 1) {
        lua_pushinteger(L, n);
        lua_rawseti(L, -2, 1);
        return 1;
    }
    int idx = 1;
    while (n % 2 == 0) {
        lua_pushinteger(L, 2);
        lua_rawseti(L, -2, idx++);
        n /= 2;
    }
    for (int64_t d = 3; d * d <= n; d += 2) {
        while (n % d == 0) {
            lua_pushinteger(L, d);
            lua_rawseti(L, -2, idx++);
            n /= d;
        }
    }
    if (n > 1) {
        lua_pushinteger(L, n);
        lua_rawseti(L, -2, idx++);
    }
    return 1;
}

static int lua_helper_comb(lua_State *L) {
    lua_Integer n = luaL_checkinteger(L, 1);
    lua_Integer k = luaL_checkinteger(L, 2);
    if (k < 0 || k > n) {
        lua_pushinteger(L, 0);
        return 1;
    }
    if (k == 0 || k == n) {
        lua_pushinteger(L, 1);
        return 1;
    }
    if (k > n / 2) {
        k = n - k;
    }
    double res = 1.0;
    for (int64_t i = 1; i <= k; i++) {
        res = res * (double)(n - i + 1) / (double)i;
    }
    double rounded = round(res);
    if (rounded <= (double)INT64_MAX) {
        lua_pushinteger(L, (int64_t)rounded);
    } else {
        lua_pushnumber(L, rounded);
    }
    return 1;
}

static int lua_helper_perm(lua_State *L) {
    lua_Integer n = luaL_checkinteger(L, 1);
    lua_Integer k = luaL_checkinteger(L, 2);
    if (k < 0 || k > n) {
        lua_pushinteger(L, 0);
        return 1;
    }
    double res = 1.0;
    for (int64_t i = 0; i < k; i++) {
        res *= (double)(n - i);
    }
    if (res <= (double)INT64_MAX) {
        lua_pushinteger(L, (int64_t)res);
    } else {
        lua_pushnumber(L, res);
    }
    return 1;
}

static int lua_helper_factorial(lua_State *L) {
    lua_Integer n = luaL_checkinteger(L, 1);
    if (n < 0) {
        lua_pushinteger(L, 0);
        return 1;
    }
    if (n > 170) {
        return luaL_error(L, "factorial overflow: %lld > 170", (long long)n);
    }
    double res = 1.0;
    for (int64_t i = 2; i <= n; i++) {
        res *= (double)i;
    }
    if (res <= (double)INT64_MAX) {
        lua_pushinteger(L, (int64_t)res);
    } else {
        lua_pushnumber(L, res);
    }
    return 1;
}

static int64_t c_extgcd(int64_t a, int64_t b, int64_t *x, int64_t *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    int64_t x1, y1;
    int64_t g = c_extgcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

static int lua_helper_modinv(lua_State *L) {
    lua_Integer a = luaL_checkinteger(L, 1);
    lua_Integer m = luaL_checkinteger(L, 2);
    if (m <= 1) {
        return luaL_error(L, "modulus must be > 1");
    }
    int64_t a_mod = (a % m + m) % m;
    int64_t x, y;
    int64_t g = c_extgcd(a_mod, m, &x, &y);
    if (g != 1) {
        return luaL_error(L, "modular inverse does not exist (gcd(%lld, %lld) != 1)", (long long)a, (long long)m);
    }
    lua_pushinteger(L, (x % m + m) % m);
    return 1;
}

// ---------------------------------------------------------
// Dynamic String Buffer for Results
// ---------------------------------------------------------

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} StrBuf;

static void buf_init(StrBuf *b) {
    b->cap = 256;
    b->len = 0;
    b->data = (char *)malloc(b->cap);
    if (b->data) b->data[0] = '\0';
}

static void buf_free(StrBuf *b) {
    if (b && b->data) {
        free(b->data);
        b->data = NULL;
    }
}

static void buf_append_str(StrBuf *b, const char *s) {
    if (!s || !b->data) return;
    size_t slen = strlen(s);
    while (b->len + slen + 1 > b->cap) {
        b->cap *= 2;
        char *nd = (char *)realloc(b->data, b->cap);
        if (!nd) return;
        b->data = nd;
    }
    memcpy(b->data + b->len, s, slen);
    b->len += slen;
    b->data[b->len] = '\0';
}

static int format_float_hd8(double fv, char *buf, size_t buf_size, char **err_out) {
    if (isnan(fv)) {
        if (err_out) *err_out = strdup("ErrDomain: NaN value cannot be bound");
        return -1;
    }
    if (isinf(fv)) {
        if (err_out) *err_out = strdup("ErrOverflow: infinite value cannot be bound");
        return -1;
    }
    if (fv == 0.0 || fabs(fv) == 0.0) {
        snprintf(buf, buf_size, "0");
        return 0;
    }

    double abs_fv = fabs(fv);

    // Integral floats with |x| <= 2^53 (9007199254740992) print as exact integers without scientific notation
    if (floor(fv) == fv && abs_fv <= 9007199254740992.0) {
        snprintf(buf, buf_size, "%.0f", fv);
        return 0;
    }

    // All other floats with |x| >= 1e15 or |x| < 1e-4 use "%.6g"
    if (abs_fv >= 1e15 || abs_fv < 1e-4) {
        snprintf(buf, buf_size, "%.6g", fv);
        return 0;
    }

    // Floats in between print with at most 6 decimal places (trailing zeros stripped)
    snprintf(buf, buf_size, "%.6f", fv);
    char *dot = strchr(buf, '.');
    if (dot) {
        char *end = buf + strlen(buf) - 1;
        while (end > dot && *end == '0') {
            *end = '\0';
            end--;
        }
        if (end == dot) {
            *end = '\0';
        }
    }
    if (strcmp(buf, "-0") == 0 || buf[0] == '\0') {
        snprintf(buf, buf_size, "0");
    }
    return 0;
}

static void format_lua_string(const char *s, size_t len, StrBuf *b) {
    buf_append_str(b, "\"");
    for (size_t i = 0; i < len; i++) {
        char c = s[i];
        if (c == '"') {
            buf_append_str(b, "\\\"");
        } else if (c == '\\') {
            buf_append_str(b, "\\\\");
        } else if (c == '\n') {
            buf_append_str(b, "\\n");
        } else if (c == '\r') {
            buf_append_str(b, "\\r");
        } else if (c == '\t') {
            buf_append_str(b, "\\t");
        } else {
            char tmp[2] = {c, '\0'};
            buf_append_str(b, tmp);
        }
    }
    buf_append_str(b, "\"");
}

static void format_lua_val(lua_State *L, int idx, StrBuf *b, char **err_out) {
    if (lua_isinteger(L, idx)) {
        lua_Integer iv = lua_tointeger(L, idx);
        char tmp[64];
        snprintf(tmp, sizeof(tmp), "%lld", (long long)iv);
        buf_append_str(b, tmp);
    } else if (lua_type(L, idx) == LUA_TNUMBER) {
        lua_Number fv = lua_tonumber(L, idx);
        char tmp[64];
        if (format_float_hd8((double)fv, tmp, sizeof(tmp), err_out) != 0) {
            return;
        }
        buf_append_str(b, tmp);
    } else if (lua_isboolean(L, idx)) {
        buf_append_str(b, lua_toboolean(L, idx) ? "true" : "false");
    } else if (lua_type(L, idx) == LUA_TSTRING) {
        size_t slen = 0;
        const char *s = lua_tolstring(L, idx, &slen);
        format_lua_string(s, slen, b);
    } else if (lua_istable(L, idx)) {
        buf_append_str(b, "[");
        lua_Integer n = luaL_len(L, idx);
        for (lua_Integer j = 1; j <= n; j++) {
            if (j > 1) buf_append_str(b, ", ");
            lua_rawgeti(L, idx, j);
            format_lua_val(L, -1, b, err_out);
            lua_pop(L, 1);
            if (err_out && *err_out != NULL) return;
        }
        buf_append_str(b, "]");
    } else {
        const char *s = luaL_tolstring(L, idx, NULL);
        buf_append_str(b, s);
        lua_pop(L, 1);
    }
}

static void remove_global(lua_State *L, const char *name) {
    lua_pushnil(L);
    lua_setglobal(L, name);
}

// ---------------------------------------------------------
// Main Bridge Entrypoint
// ---------------------------------------------------------

int run_lua_sandboxed(
    const char *script,
    int reg_count,
    const char **reg_names,
    const char **reg_vals,
    const int *reg_types,
    char **out_result,
    char **out_err
) {
    *out_result = NULL;
    *out_err = NULL;

    MemLimiter ml;
    ml.used_bytes = 0;
    ml.max_bytes = MAX_HEAP_BYTES;

    lua_State *L = lua_newstate(sandboxed_alloc, &ml);
    if (!L) {
        *out_err = strdup("ErrOverflow: memory limit");
        return -1;
    }

    // Open safe base libraries
    luaL_requiref(L, "_G", luaopen_base, 1);
    lua_pop(L, 1);
    luaL_requiref(L, LUA_TABLIBNAME, luaopen_table, 1);
    lua_pop(L, 1);
    luaL_requiref(L, LUA_STRLIBNAME, luaopen_string, 1);
    lua_pop(L, 1);
    luaL_requiref(L, LUA_MATHLIBNAME, luaopen_math, 1);
    lua_pop(L, 1);
    luaL_requiref(L, LUA_UTF8LIBNAME, luaopen_utf8, 1);
    lua_pop(L, 1);

    // Strip unsafe global functions
    remove_global(L, "pcall");
    remove_global(L, "xpcall");
    remove_global(L, "load");
    remove_global(L, "loadstring");
    remove_global(L, "loadfile");
    remove_global(L, "dofile");
    remove_global(L, "require");
    remove_global(L, "collectgarbage");
    remove_global(L, "rawget");
    remove_global(L, "rawset");
    remove_global(L, "rawequal");

    // Replace print with dummy
    lua_pushcfunction(L, dummy_print);
    lua_setglobal(L, "print");

    // Deterministically seed math.random
    lua_getglobal(L, "math");
    if (lua_istable(L, -1)) {
        lua_getfield(L, -1, "randomseed");
        if (lua_isfunction(L, -1)) {
            lua_pushinteger(L, 42);
            lua_pushinteger(L, 42);
            lua_pcall(L, 2, 0, 0);
        } else {
            lua_pop(L, 1);
        }
    }
    lua_pop(L, 1);

    // Strip string.dump
    lua_getglobal(L, "string");
    if (lua_istable(L, -1)) {
        lua_pushnil(L);
        lua_setfield(L, -2, "dump");
    }
    lua_pop(L, 1);

    // Protect string metatable
    lua_pushstring(L, "");
    if (lua_getmetatable(L, -1)) {
        lua_pushboolean(L, 0);
        lua_setfield(L, -2, "__metatable");
        lua_pop(L, 1);
    }
    lua_pop(L, 1);

    // Register mathematical helpers
    lua_register(L, "gcd", lua_helper_gcd);
    lua_register(L, "lcm", lua_helper_lcm);
    lua_register(L, "isprime", lua_helper_isprime);
    lua_register(L, "factors", lua_helper_factors);
    lua_register(L, "comb", lua_helper_comb);
    lua_register(L, "perm", lua_helper_perm);
    lua_register(L, "factorial", lua_helper_factorial);
    lua_register(L, "modinv", lua_helper_modinv);

    // Inject registers
    for (int i = 0; i < reg_count; i++) {
        const char *name = reg_names[i];
        const char *val = reg_vals[i];
        int type = reg_types ? reg_types[i] : REG_TYPE_AUTO;
        if (!name || !val) continue;

        if (type == REG_TYPE_BOOL) {
            int b = (strcmp(val, "true") == 0);
            lua_pushboolean(L, b);
            lua_setglobal(L, name);
            continue;
        }
        if (type == REG_TYPE_STRING) {
            lua_pushstring(L, val);
            lua_setglobal(L, name);
            continue;
        }
        if (type == REG_TYPE_INT) {
            char *endptr = NULL;
            errno = 0;
            long long iv = strtoll(val, &endptr, 10);
            if (errno == 0 && *endptr == '\0') {
                lua_pushinteger(L, (lua_Integer)iv);
                lua_setglobal(L, name);
                continue;
            }
        } else if (type == REG_TYPE_FLOAT) {
            char *endptr = NULL;
            errno = 0;
            double fv = strtod(val, &endptr);
            if (errno == 0 && *endptr == '\0') {
                lua_pushnumber(L, (lua_Number)fv);
                lua_setglobal(L, name);
                continue;
            }
        }

        // Fallback auto-detection if type == REG_TYPE_AUTO
        if (strcmp(val, "true") == 0) {
            lua_pushboolean(L, 1);
            lua_setglobal(L, name);
            continue;
        }
        if (strcmp(val, "false") == 0) {
            lua_pushboolean(L, 0);
            lua_setglobal(L, name);
            continue;
        }

        char *endptr = NULL;
        errno = 0;
        long long iv = strtoll(val, &endptr, 10);
        if (*val != '\0' && *endptr == '\0' && errno == 0) {
            lua_pushinteger(L, (lua_Integer)iv);
            lua_setglobal(L, name);
        } else {
            errno = 0;
            double fv = strtod(val, &endptr);
            if (*val != '\0' && *endptr == '\0' && errno == 0) {
                lua_pushnumber(L, (lua_Number)fv);
                lua_setglobal(L, name);
            } else {
                lua_pushstring(L, val);
                lua_setglobal(L, name);
            }
        }
    }

    // Set instruction hook with 20M VM instructions and 1s wall-clock backstop
    tl_instruction_count = 0;
    tl_deadline_ns = get_time_ns() + (uint64_t)TIMEOUT_WALLCLOCK_MS * 1000000ULL;
    lua_sethook(L, timeout_hook, LUA_MASKCOUNT, HOOK_STEP);

    // Compile script
    int load_status = luaL_loadstring(L, script);
    if (load_status != LUA_OK) {
        size_t elen = 0;
        const char *err_msg = luaL_tolstring(L, -1, &elen);
        *out_err = strdup(err_msg ? err_msg : "syntax error");
        tl_deadline_ns = 0;
        tl_instruction_count = 0;
        lua_close(L);
        return -1;
    }

    // Execute script
    int run_status = lua_pcall(L, 0, LUA_MULTRET, 0);
    tl_deadline_ns = 0; // Clear deadline
    tl_instruction_count = 0;

    if (run_status != LUA_OK) {
        size_t elen = 0;
        const char *err_msg = luaL_tolstring(L, -1, &elen);
        *out_err = strdup(err_msg ? err_msg : "runtime error");
        lua_close(L);
        return -1;
    }

    int top = lua_gettop(L);
    if (top == 0) {
        *out_err = strdup("script returned no values; scripts must terminate with 'return answer'");
        lua_close(L);
        return -1;
    }

    StrBuf sb;
    buf_init(&sb);

    for (int i = 1; i <= top; i++) {
        if (i > 1) buf_append_str(&sb, ", ");
        format_lua_val(L, i, &sb, out_err);
        if (*out_err != NULL) {
            buf_free(&sb);
            lua_close(L);
            return -1;
        }
    }

    lua_close(L);

    *out_result = sb.data;
    return 0;
}

void free_lua_string(char *s) {
    if (s) free(s);
}
