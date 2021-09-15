# RSL Transpiler Architecture
---
### Stage 1: AST Parser
Each .rsl file is parsed into an abstract syntax tree.

| AST Component | Structure |
|---|---|
| SyntaxTree | `[sourceentry]` |
| Node | `|comment, import, struct, function|` |
| Type | `|all supported types|` |
| Dependency | `{ident, scope}` |
| Scope | { ident, || module{ident}, common || }
| Struct | { ident, public, [entry], attribute? }
| Entry | { ident, type, location?, builtin? }
| Attribute | || vertex, transport, uniform, custom{ident} ||
| Function | { ident, public, [arg], output, block }
| Arg | { ident, type, attribute? }
| Statement | || declaration, naked{expression} ||
| Declaration | { ident, type, expression }
| Expression | || line{string}, bloc ||
| Bloc | || if, loop ||
| If | { expression, block }
| Loop | { ident, iteration, block }
| Iteration | || TODO range{u32, u32}, TODO array{ident}, infinite ||
| Block | { [statement], expression? }
| Comment | { string }

---
### Stage 2: Linker
1.  The vertex/fragment syntax trees in each module are merged.
    (Shared imports are moved up)
3.  Any dependencies needed by vert from frag are moved up.
4.  Imports are cloned into each module ast.

Each rsl module tree is now finalized.

---
### Stage 3: Rust Generator

1. Structs with the vertex attribute are used to generate wgsl-rust.
2. Structs with the uniform attribute are used to generate wgsl-rust.

The rsl module trees are now ready to be transpiled into wgsl.

---
### Stage 4: WGSL Generator

RSL &rarr; WGSL Modes


`inherited` - independently transpiled subcomponents templated together
`computed` - wgsl generated from subcomponents
`mapped` - one-to-one mapping of rsl -> wgsl (e.g. types)
`inlined` - no transformations

Components consumed only by previous stages are marked as n/a.

 AST Component       Mode
 -----------------------------------------------------
 - syntaxtree        inherited
 - node              inherited

 - type              mapped
 - import            n/a
 - scope             n/a
 - struct            computed (mut: transport layouts, builtin attrs, [[block]] uniforms)
 - entry             inherited
 - attribute         n/a
 - function          computed (mut: uniform args)
 - arg               inherited
 - statement         inherited
 - declaration       inherited
 - expression        computed (mut: struct constructors)
 - bloc              inherited
 - if                inherited
 - loop              inherited
 - iteration         computed (mut: todo)
 - block             inherited
 - comment           inlined