# Naming conventions

## Symbol naming

| Symbol kind           | Style               | Example        |
|----------------------|---------------------|-----------------|
| Macros                | `UPPER_SNAKE_CASE`  | `MY_MACRO`      |
| Template parameters   | `CamelCase`         | `MyParameter`   |
| All other symbols     | `lower_snake_case`  | `my_variable`   |

## Non-public symbols — reserved identifier prefix

Non-public symbols must be C++ reserved identifiers:

- Macros and template parameters: single-underscore prefix — `_MY_MACRO`, `_MyParameter`.
- All other symbols: double-underscore prefix — `__my_variable`.

## Template parameter names

Avoid single-letter template parameter names. Prefer short but descriptive names:

- Wrong: `_T`, `_U`
- Correct: `_Tp`, `_Up`, `_Key`, `_Value`

## Type qualification

Type names must be fully qualified except when already declared in the current namespace.
This includes standard integer type aliases:

```cpp
::cuda::std::size_t
::cuda::std::uintptr_t
::cuda::std::int32_t
```

A local `using` declaration is acceptable to reduce repetition within a function body:

```cpp
using ::cuda::std::size_t;
```

## Free function calls

All calls to free functions must be fully qualified from the global namespace — including
calls to functions in the same namespace:

```cpp
// Inside namespace cuda::
::cuda::ceil_div(a, b);   // correct
ceil_div(a, b);            // wrong — unqualified
```

This rule does not apply to static member functions of classes.
