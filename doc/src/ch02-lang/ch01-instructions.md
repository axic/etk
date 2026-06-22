# Instructions

Instructions, also known as opcodes or `Op`s internally, are the building blocks of ETK smart contracts. Each instruction has a human-readable mnemonic (like `dup3`) and the machine readable equivalent (which would be `0x82`). The `push` family of instructions also encode an immediate value (or argument.)


## List of Instructions

```ignore
{{#include ../../../etk-asm/tests/asm/every-op/main.etk}}
```

## Directives

### `.org <offset>`

Sets the base offset for the code that follows. The assembler pads the output with zero bytes from the current position up to `<offset>`. Labels defined after `.org <offset>` will resolve to `<offset> + N`, where `N` is the number of bytes since the directive.

Multiple `.org` directives are allowed; each updates the base offset for the code following it. The requested offset must be greater than or equal to the assembler's current position — rewinding is an error.

```rust
# extern crate etk_asm;
# let src = r#"
push1 1
.org 0x10
lbl:
    jumpdest
# "#;
# let mut output = Vec::new();
# let mut ingest = etk_asm::ingest::Ingest::new(&mut output);
# ingest.ingest(file!(), src).unwrap();
# assert_eq!(output.len(), 0x11);
# assert_eq!(output[0], 0x60);
# assert_eq!(output[1], 0x01);
# assert_eq!(output[0x10], 0x5b);
```
